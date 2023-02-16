from operator import mod
from tkinter import Y
from trace import Trace

from numpy import angle
import os
import json
import numpy as np
import random
import time
import sys
import math
from collections import deque

import torch
import torch.nn as nn
# from agent.net import Net
# from agent.network import Net
from agent.network_v1 import Net
# from agent.network_v0 import Net
from torch.utils.tensorboard import SummaryWriter
import pdb
from matplotlib import pyplot as plt
from utils.geometry import points2line

class Agent(nn.Module):
    def __init__(self, state_size, action_size, config=None, mode = 'train', device = torch.device('cpu')):
        super().__init__()
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000 # maximum steps per episode
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05

        self.device = device
        self.mode = mode
        print(f'agent training device is loaded: {self.device}')

        self.lr = 1e-3
        self.length = self.config.Model.length

        # trajectory generation
        self.plan_net = Net(self.state_size, self.action_size, length = self.length, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.plan_net.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.plan_net.parameters(),lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

        # loss weight
        self.dist_weight = 1.
        self.close_weight = 2.
        self.col_weight = 1.
        self.smooth_weight = 0.01
        self.step_weight = 0.01
        self.bspline_weight = 0.01
        self.total_loss = []

        # adaptive weight on loss weight
        self.alpha_dist = torch.tensor((0.5), requires_grad=True).type(torch.FloatTensor).to(self.device)

        # smooth epsilon, prevent backward nan
        self.sm_epsilon = 1e-8

        # cbf loss
        self.cbf_epsilon = 0.5
        self.safe_dist = 0.075
        self.act_cbf = nn.ReLU()

        # clf loss
        self.clf_epsilon = 1.
        self.act_clf = nn.ReLU()

        # entropy&unobservable distance loss
        self.r2g_safe = 0.075
        self.obs_loss = True
        self.obs_weight = 0.5

    def path_planning(self, x):
        """
        Description:
        args:
            x: numpy, (state_size,)n
        return:
            traj: (l,2), (phi,theta)
        """
        # pdb.set_trace()
        # import ipdb;ipdb.set_trace()
        # print(x.shape)
        # x = torch.tensor(x,dtype=torch.float,device=self.device).unsqueeze(0) #(1, state_size)


        traj = self.plan_net(x, self.mode)

        return traj
    
    def traj_follower(self, traj):
        # print(f'shape:{traj.shape}')
        # print(traj[1])
        angular = np.pi / 2 - traj[1,1].detach().numpy()

        linear = np.clip(0.2 * traj[1,0].detach().numpy() / 0.015, a_min = 0., a_max = 0.2)

        return linear, angular

    def learn(self, traj, data, train=True):
        """calculate self-supervised learning loss
        Args:
            traj (b,l,2): _description_
            data (b,370): lidar_r(0:360),angular(361:362),goal_dir(363:364),init_w(364:365),cur_pose_w(366:367),goal_pose_r(368:369)
        """
        loss = 0

        # evaluate distance to goal position
        # print(traj)
        # x = torch.from_numpy(x)

        # pdb.set_trace()
        # if x.shape[0] != self.batch_size:
        # pdb.set_trace()
        self.batch_size = traj.shape[0]

        rad_points = torch.zeros((self.batch_size, 360, 2),device=self.device)
        # pdb.set_trace()
        rad_points[:,:,0] = torch.cos((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]
        rad_points[:,:,1] = torch.sin((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]

        # trajectory distance loss
        loss_dist = 0
        ini_dist = torch.linalg.norm(data[:,-2:], dim=-1)
        tgt_dist = torch.linalg.norm(traj[:,-1,:] - data[:,-2:], dim=-1)

        # mask out distances that are larger than 3.5m
        init_mask = torch.lt(ini_dist,3.5)  #(b,)
        # loss_dist = -(ini_dist - tgt_dist) * init_mask * (1/ini_dist)
        # loss_dist -= (ini_dist - tgt_dist) * torch.logical_not(init_mask) /3.5

        loss_dist = tgt_dist * init_mask * (1/ini_dist)
        # loss_dist = (tgt_dist * (1/ini_dist))[init_mask].sum()
        # loss_dist += ((3.5 - (ini_dist - tgt_dist))/3.5) * torch.logical_not(init_mask)
        loss_dist += ((3.5 - torch.clamp((ini_dist - tgt_dist),max=3.5))/3.5) * torch.logical_not(init_mask)
        # loss_dist += ((3.5 - torch.clamp((ini_dist - tgt_dist),max=3.5))/3.5)[torch.logical_not(init_mask)].sum()
        # loss_dist = loss_dist / self.batch_size
        loss_dist = loss_dist.mean()

        # loss_dist = -(ini_dist - tgt_dist).mean()
        if not(train):
            print(f'current distance loss: {-(ini_dist - tgt_dist)[0]}, maximum: {-ini_dist[0]}')

        # pdb.set_trace()

        # # temporal clf loss design
        # # pdb.set_trace()
        # loss_dist = 0.
        # pre_dist = torch.linalg.norm(data[:,-2:],dim=-1)  #(b,)
        # v_pre_dist = pre_dist ** 2
        # cur_dist = torch.linalg.norm(traj[:,-1,:] - data[:,-2:], dim = -1)  #(b,)
        # v_cur_dist = cur_dist ** 2
        # loss_dist = self.act_clf(v_cur_dist - v_pre_dist + self.clf_epsilon * v_cur_dist).mean()

        # # clf loss design
        # loss_dist = 0.
        # pre_dist = torch.linalg.norm(data[:,-2:],dim=-1)  #(b,)
        # for i in range(self.length):
        #     cur_dist = torch.linalg.norm(traj[:,-1,:] - data[:,-2:], dim = -1)  #(b,)
        #     loss_dist += self.act_clf(cur_dist - pre_dist + self.clf_epsilon * cur_dist).mean()
        #     pre_dist = cur_dist
        # loss_dist = loss_dist / self.length

        # entropy&unobservable distance loss, if traj not arrived at goal point, then it should point to goal positions without any obstacles between them
        # pdb.set_trace()
        if self.obs_loss:
            r2g = points2line(rad_points,traj[:,-1,:],data[:,-2:])  #(b,360)
            r2g_min = r2g.min(-1).values  #(b,)

            # goal mask to calculate distance between last predictive waypoint to goal position if not reached
            goal_mask = torch.linalg.norm(traj[:,-1,:] - data[:,-2:], dim=-1)  #(b,)
            goal_mask = torch.gt(goal_mask,0.01)
            # pdb.set_trace()
            # test = torch.clamp((self.r2g_safe - r2g_min) / self.r2g_safe, min = 0., max = 1.)[goal_mask]
            loss_dist += self.obs_weight * torch.clamp((self.r2g_safe - r2g_min) / self.r2g_safe, min = 0., max = 1.)[goal_mask].sum()/self.batch_size

        # ini_dist = torch.linalg.norm(data[:,-2:], dim=-1)  #(b,)
        # tgt_dist = torch.linalg.norm(traj[:,-1,:] - data[:,-2:], dim=-1)  #(b,)

        # if torch.gt(r2g_min,self.r2g_safe).sum() > 0:
        #     loss_dist += ((tgt_dist - ini_dist) / ini_dist)[torch.gt(r2g_min, self.r2g_safe)].mean() * self.close_weight
        
        # loss_dist = torch.exp(r2g_loss) * (tgt_dist - ini_dist) / ini_dist)
        # loss_dist = torch.gt(r2g_min,self.r2g_safe).detach() * torch.log(tgt_dist / ini_dist + 1e-6)
        # loss_dist = torch.log(tgt_dist / ini_dist + 1e-6)[torch.gt(r2g_min, self.r2g_safe)]
        # loss_dist += r2g_loss

        # ini_dist = torch.linalg.norm(data[:,-2:], dim=-1)
        # for i in range(self.length):
        #     tgt_dist.append(torch.linalg.norm(traj[:,i,:] - data[:,-2:], dim=-1))
        #     # if goal reached:
        #     goal_reached_mask = torch.lt(tgt_dist[-1], 0.2).detach()  #(b,)
        #     goal_not_reached_mask = torch.logical_not(goal_reached_mask)  #(b,)
        #     # loss_dist -= (0.2 - tgt_dist[i]) * goal_reached_mask

        #     # else goal not reached
        #     if i == self.length - 1:
        #         loss_dist -= (torch.linalg.norm(data[:,-2:],dim=-1) - tgt_dist[i]) * goal_not_reached_mask
        #     # if i == 0:
        #     #     loss_dist -= (ini_dist - tgt_dist[i]) * goal_not_reached_mask
        #     # else:
        #     #     loss_dist -= (tgt_dist[i-1] - tgt_dist[i]) * goal_not_reached_mask
        
        # step size loss, to encourage each step to take as large as it can
        # pdb.set_trace()
        # loss_step = 0.
        # step_size = []
        # step_size.append(traj[:,0,:])
        # loss_step += (0.3 * np.sqrt(2) - torch.linalg.norm(step_size[-1], dim=-1)).mean()
        # for i in range(1,self.length):
        #     step_size.append(traj[:,i,:] - traj[:,i-1,:])
        #     loss_step += (0.3 * np.sqrt(2) - torch.linalg.norm(step_size[-1], dim=-1)).mean()
        #     if (0.3 * np.sqrt(2) + 0.01 - torch.linalg.norm(step_size[-1], dim=-1)).mean() <= 0:
        #         pdb.set_trace()
        
        # # collision avoidance loss
        # # pdb.set_trace()
        # loss_col = 0
        # for i in range(1, self.length):
        #     err_x = traj[:,i,0][:,None] - rad_points[:,:,0]  #(b,360)
        #     err_y = traj[:,i,1][:,None] - rad_points[:,:,1]  #(b,360)
        #     err_dist = torch.sqrt(err_x**2 + err_y**2).min(-1).values  #(b,)
        #     # loss_col += torch.clamp((0.15-err_dist)/0.15, min = 0., max = 1.).mean()
        #     # if distance is less than 0.10, then it should be penalized largely, otherwise it could be compromised by other loss
        #     collision_loss = torch.clamp(0.15 - err_dist, min = 0., max = 0.05)  #(0.,0.05)
        #     collision_loss += collision_loss * (collision_loss == 0.05) * 19  #(if equals to 0.05, +0.95) (range:[0:0.05,1.])

        #     loss_col += collision_loss.mean()

        # # control barrier function loss with point2line distance
        # loss_col = 0
        # # pdb.set_trace()
        # init_points = torch.tensor([0.,0.]).to(self.device)[None,:].repeat(self.batch_size,1)  #(b,2)
        # # pre_dist = points2line(rad_points,traj[:,0,:],init_points).min(-1).values - self.safe_dist  #(b,)
        # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # err_dist = torch.cdist(traj, rad_points).min(-1).values - self.safe_dist  #(b,l)
        # for i in range(0, self.length):
        #     if i == 0:
        #         err_dist = points2line(rad_points,traj[:,i,:],init_points).min(-1).values - self.safe_dist #(b,)
        #         loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()
        #         pre_dist = err_dist
        #     else:
        #         err_dist = points2line(rad_points,traj[:,i,:],traj[:,i-1,:]).min(-1).values - self.safe_dist #(b,)
        #         loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()
        #         pre_dist = err_dist
        
        # control barrier function loss with point2line distance by iteratively detaching previous one
        loss_col = 0
        # pdb.set_trace()
        init_points = torch.tensor([0.,0.]).to(self.device)[None,:].repeat(self.batch_size,1)  #(b,2)
        # pre_dist = points2line(rad_points,traj[:,0,:],init_points).min(-1).values - self.safe_dist  #(b,)
        pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # err_dist = torch.cdist(traj, rad_points).min(-1).values - self.safe_dist  #(b,l)
        for i in range(0, self.length):
            if i == 0:
                err_dist = points2line(rad_points,traj[:,i,:],init_points).min(-1).values - self.safe_dist #(b,)
                loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()
                pre_dist = err_dist
            else:
                err_dist = points2line(rad_points,traj[:,i,:],traj[:,i-1,:].detach()).min(-1).values - self.safe_dist #(b,)
                loss_col += self.act_cbf(-(err_dist - pre_dist.detach() + self.cbf_epsilon * err_dist)).mean()
                pre_dist = err_dist
        
        # # control barrier function loss with point2line distance by iteratively detaching previous one and step function
        # loss_col = 0
        # # pdb.set_trace()
        # init_points = torch.tensor([0.,0.]).to(self.device)[None,:].repeat(self.batch_size,1)  #(b,2)
        # # pre_dist = points2line(rad_points,traj[:,0,:],init_points).min(-1).values - self.safe_dist  #(b,)
        # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # err_dist = torch.cdist(traj, rad_points).min(-1).values - self.safe_dist  #(b,l)
        # for i in range(0, self.length):
        #     if i == 0:
        #         err_dist = points2line(rad_points,traj[:,i,:],init_points).min(-1).values - self.safe_dist #(b,)
        #         temp_loss = self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist))
        #         temp_loss[temp_loss>0] = 1
        #         temp_loss[temp_loss<=0] = 0
        #         # loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()
        #         loss_col += temp_loss.mean()
        #         pre_dist = err_dist
        #     else:
        #         err_dist = points2line(rad_points,traj[:,i,:],traj[:,i-1,:].detach()).min(-1).values - self.safe_dist #(b,)
        #         # loss_col += self.act_cbf(-(err_dist - pre_dist.detach() + self.cbf_epsilon * err_dist)).mean()
        #         temp_loss = self.act_cbf(-(err_dist - pre_dist.detach() + self.cbf_epsilon * err_dist))
        #         temp_loss[temp_loss>0] = 1
        #         temp_loss[temp_loss<=0] = 0
        #         loss_col += temp_loss.mean()
        #         pre_dist = err_dist

        # # temporal control barrier function
        # loss_col = 0
        # # pdb.set_trace()
        # init_points = torch.tensor([0.,0.]).to(self.device)[None,:].repeat(self.batch_size,1)  #(b,2)
        # # pre_dist = points2line(rad_points,traj[:,0,:],init_points).min(-1).values - self.safe_dist  #(b,)
        # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # # err_dist = torch.cdist(traj, rad_points).min(-1).values - self.safe_dist  #(b,l)
        # for i in range(0, self.length):
        #     if i == 0:
        #         err_dist = points2line(rad_points,traj[:,i,:],init_points).min(-1).values - self.safe_dist #(b,)
        #         loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()
        #     else:
        #         err_dist = points2line(rad_points,traj[:,i,:],traj[:,i-1,:]).min(-1).values - self.safe_dist #(b,)
        #         loss_col += self.act_cbf(-(err_dist - pre_dist + self.cbf_epsilon * err_dist)).mean()

        # # control barrier function loss for safety without point2line distance
        # loss_col = 0
        # pre_dist = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)
        # err_dist = torch.cdist(traj, rad_points).min(-1).values - self.safe_dist  #(b,l)
        # for i in range(0, self.length):
        #     if i == 0:
        #         loss_col += self.act_cbf(-(err_dist[:,i] - pre_dist + self.cbf_epsilon * err_dist[:,i])).mean()
        #     else:
        #         loss_col += self.act_cbf(-(err_dist[:,i] - err_dist[:,i-1] + self.cbf_epsilon * err_dist[:,i])).mean()

        # evaluate trajectory smoothness
        angle_loss = 0.
        for i in range(1,self.length):
            # angle_loss = torch.abs(torch.arctan((traj[:,i,1] - traj[:,i-1,1])/(traj[:,i,0] - traj[:,i-1,0]))).mean()
            # angle_loss += torch.abs(torch.arctan(traj[:,i,1]/traj[:,i,0]) -  torch.arctan(traj[:,i-1,1]/traj[:,i-1,0])).mean()
            # angle_loss += torch.abs(torch.atan2(traj[:,i,1],traj[:,i,0]) -  torch.atan2(traj[:,i-1,1],traj[:,i-1,0])).mean()
            # angle_loss += torch.abs(torch.atan2(traj[:,i,1]+self.sm_epsilon,traj[:,i,0]+self.sm_epsilon) -  torch.atan2(traj[:,i-1,1]+self.sm_epsilon,traj[:,i-1,0]+self.sm_epsilon)).mean()
            angle_loss += torch.abs(torch.atan2(traj[:,i,1], traj[:,i,0] + self.sm_epsilon) -  torch.atan2(traj[:,i-1,1], traj[:,i-1,0] + self.sm_epsilon)).mean()
            # angle_loss += torch.abs(traj[i][:,1] - traj[i-1][:,1]).mean()
        # angle_loss = torch.abs(traj[:,1:,1] - traj[:,0:-1,1]).sum(-1)

        # trajectory bspline jerk smoothness loss
        bspline_loss = 0.
        for i in range(self.length - 3):
            jerk = traj[:,i+3] - 3 * traj[:,i+2] + 3 * traj[:,i+1] - traj[:,i]
            # bspline_loss += torch.sqrt(jerk ** 2).mean()
            bspline_loss += (jerk ** 2).mean()

        # total loss
        # pdb.set_trace()
        # loss = self.col_weight * loss_col + self.smooth_weight * angle_loss
        # loss = self.dist_weight * loss_dist.mean() + self.col_weight * loss_col + self.smooth_weight * angle_loss
        distance_loss = self.dist_weight * loss_dist
        collision_loss  = self.col_weight * loss_col
        smooth_loss = self.smooth_weight * angle_loss
        bspline_loss = self.bspline_weight * bspline_loss

        # loss = distance_loss
        # loss = loss_dist + loss_col
        # loss = distance_loss + collision_loss
        # loss = distance_loss + collision_loss + smooth_loss
        loss = distance_loss + collision_loss + bspline_loss

        # # adaptive weighted loss
        # loss = self.alpha_dist * distance_loss + (1. - self.alpha_dist) * collision_loss + smooth_loss

        # pdb.set_trace()
        if not(train):
            plt.scatter(rad_points[0,:,0].cpu().numpy(),rad_points[0,:,1].cpu().numpy())
            
            plt.scatter([0.], [0.], color='black')
            for i in range(self.length):
                plt.scatter(traj[0,i,0].detach().cpu().numpy(), traj[0,i,1].detach().cpu().numpy(), color='green')
            
            # pdb.set_trace()
            plt.scatter(data[0,-2].cpu().numpy(), data[0,-1].cpu().numpy(),color='red')
            plt.show()

        # print(f'current loss: {loss:3f}||dist_loss:{self.dist_weight * loss_dist.mean():3f}||collision:{self.col_weight * loss_col:3f}||angle:{self.smooth_weight * angle_loss.mean():3f}')

        # calculate local gradients and push local parameters to global
        if train:
            with torch.autograd.detect_anomaly():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

        return loss.detach().cpu().numpy(), distance_loss.detach().cpu().numpy(), collision_loss.detach().cpu().numpy(), smooth_loss.detach().cpu().numpy(), bspline_loss.detach().cpu().numpy()