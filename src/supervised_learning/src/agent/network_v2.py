import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np
from collections import namedtuple,deque
import torch.multiprocessing as mp
import pdb
from agent.attention import Crossformer
from utils.geometry import points2line

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, scan_range = 360, length = 10, device = torch.device('cpu')):
        super(Net, self).__init__()
        self.length = length
        self.device = device
        
        self.feats_dim = 64
        self.heads = 8
        self.safe_dist = 0.075
        
        self.backbone = nn.Sequential(
                nn.Linear(scan_range, 360),
                nn.ReLU(),
                nn.Linear(360, self.feats_dim),
                nn.ReLU()).to(device)

        self.act_cbf = nn.ReLU()
        self.cbf_epsilon = 0.5
        self.lr = 1e-2
        self.momentum = 0.5
        self.num_steps = 1
        
        self.plan = nn.Sequential(
                    nn.Linear(self.feats_dim + 4, 128),
                    nn.ReLU(),
                    nn.Linear(128,  self.feats_dim),
                    nn.ReLU()).to(device)

        self.plan_decoder = nn.Sequential(
                    nn.Linear(self.feats_dim, self.feats_dim),
                    nn.ReLU(),
                    nn.Linear(self.feats_dim,  2)).to(device)

        self.traj_decoder = nn.Sequential(
                    nn.Linear(self.feats_dim + 4, 128),
                    nn.ReLU(),
                    nn.Linear(128,  self.feats_dim),
                    nn.ReLU()).to(device)
        
        self.norm1 = nn.BatchNorm1d(2)
        
        self.traj_mlp = nn.Sequential(
                nn.Linear(self.feats_dim, 2)).to(device)
        
        self.use_att = True

        if self.use_att:
            self.traj = nn.Sequential(
                    nn.Linear(self.feats_dim + 2, 512),
                    nn.ReLU(),
                    nn.Linear(512,  self.length * self.feats_dim)).to(device)
        else:
            self.traj = nn.Sequential(
                    nn.Linear(self.feats_dim + 2, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128,  self.length * 2)).to(device)
        
        self.crossformer = Crossformer(self.feats_dim, self.heads, self.feats_dim, self.feats_dim, device=self.device)

        self.linear_constraint = 0.22
        self.omega_constraint = 0.5

        # self.apply(weights_init_)
    
    def ineq_partial_steps(self, h_x, h_x_1, h_x_2 = None):
        g_x = h_x - h_x_1 + self.cbf_epsilon * h_x
        delta_g_x = h_x + h_x_2 - 2 * h_x_1 + self.cbf_epsilon * (h_x - h_x_1)
        return 2 * self.act_cbf(-g_x) * (-delta_g_x)
    
    def single_step_ineq_corr(self, rad_points, cur_pos):
        """correct traj waypoints to fulfill cbf inequality constraint

        Args:
            rad_points (tensor.float): (b,360)
            cur_pos (tensor.float): (b,15,2)
        Return:
            Y_step: (tensor.float), (b,15)
        """
        # pdb.set_trace()
        b = rad_points.shape[0]
        init_points = torch.tensor([0.,0.]).to(self.device)[None,:].repeat(b,1)  #(b,2)

        h_x_1 = torch.linalg.norm(rad_points, dim=-1).min(-1).values - self.safe_dist  #(b,)

        for i in range(0, self.length):
            if i == 0:
                h_x_2 = points2line(rad_points,cur_pos[:,i,:],init_points).min(-1).values - self.safe_dist  #(b,)
                Y_step = 2 * self.act_cbf(-(h_x_2 - h_x_1 + self.cbf_epsilon * h_x_2)) * (h_x_2 - 2 * h_x_1 + self.cbf_epsilon * (h_x_2 - h_x_1))  #(b,)
                Y_step = Y_step[:,None]  #(b,1)
                h_x_0 = h_x_1
                h_x_1 = h_x_2
            else:
                h_x_2 = points2line(rad_points,cur_pos[:,i,:],cur_pos[:,i-1,:].detach()).min(-1).values - self.safe_dist  #(b,)
                Y_temp_step = 2 * self.act_cbf(-(h_x_2 - h_x_1 + self.cbf_epsilon * h_x_2)) * (h_x_2 + h_x_0 - 2 * h_x_1 + self.cbf_epsilon * (h_x_2 - h_x_1))
                Y_step = torch.cat([Y_step, Y_temp_step[:,None]], dim=-1)  #(b,i)
                h_x_0 = h_x_1
                h_x_1 = h_x_2
        
        return Y_step

    def ineq_corr(self, rad_points, cur_pos, lr, momentum, num_steps):
        """take multi steps to refine and correct inequality constraint

        Args:
            rad_points (_type_): _description_
            cur_pos (_type_): _description_
            lr (_type_): _description_
            momentum (_type_): _description_
            num_steps (_type_): _description_
        """
        old_Y_step = None
        for i in range(num_steps):
            Y_step = self.single_step_ineq_corr(rad_points, cur_pos)  #(b,15)
            if i == 0:
                new_Y_step = lr * Y_step
                old_Y_step = new_Y_step
            else:
                new_Y_step = lr * Y_step + momentum * old_Y_step
            
            cur_pos = cur_pos - new_Y_step[:,:,None]
        
        return cur_pos

    def forward(self, s, mode='train'):
        """
        Description: actor policy
        args:
            s: (batch_size, state_dim: 372)
        return:
            a: [length](batch_size, 2)
        """
        b = s.shape[0]
        rad_points = torch.zeros((b, 360, 2),device=self.device)  #(b,360,2)
        # pdb.set_trace()
        rad_points[:,:,0] = torch.cos((torch.arange(0,360).to(self.device)) * torch.pi / 180) * s[:,0:360]
        rad_points[:,:,1] = torch.sin((torch.arange(0,360).to(self.device)) * torch.pi / 180) * s[:,0:360]

        if mode == 'train':
            scan_embedding = self.backbone(s[:,:360])  #(b,d)
            # scan_embedding = self.backbone_v0(rad_points.view(b,-1))  #(b,d)
            
            # # without recursive
            total_embedding = torch.cat([scan_embedding, s[:,-2:]], dim=1)
            if self.use_att:
                traj_embedding = self.traj(total_embedding).view(b,-1,self.feats_dim)  #(b,l,d)
            else:
                cur_pos = self.traj(total_embedding).view(b,self.length,2)  #(b,l,2)
                for i in range(0,self.length):
                    if i == 0:
                        cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i])  #(b,l,2)
                    else:
                        cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i]) + cur_pos[:,i-1]

            # attention module
            if self.use_att:
                # cur_pos = self.crossformer(cur_pos, s[:,:360])  #(b,l,d)
                # cur_pos = self.crossformer(traj_embedding, s[:,:360])  #(b,l,d)
                cur_pos = self.crossformer(traj_embedding, s[:,:360], s[:,-2:])  #(b,l,d)
                cur_pos = self.traj_mlp(cur_pos)  #(b,l,2)
                # cur_pos = 0.3 * torch.tanh(cur_pos)  #(b,l,2)

            # omega_data
            omega_data = None
            if self.use_att:
                for i in range(0,self.length):
                    if i == 0:
                        linear_delta = self.linear_constraint * torch.tanh(cur_pos[:,i,0])  #(b,)
                        omega_delta  = self.omega_constraint  * torch.tanh(cur_pos[:,i,1])  #(b,)
                        cur_pos[:,i,0] = linear_delta * torch.cos(omega_delta)
                        cur_pos[:,i,1] = linear_delta * torch.sin(omega_delta)
                    else:
                        linear_delta = self.linear_constraint * torch.tanh(cur_pos[:,i,0])  #(b,)
                        omega_delta  = self.omega_constraint  * torch.tanh(cur_pos[:,i,1])  #(b,)
                        cur_pos[:,i,0] = linear_delta * torch.cos(omega_delta) + cur_pos[:,i-1,0]
                        cur_pos[:,i,1] = linear_delta * torch.sin(omega_delta) + cur_pos[:,i-1,1]

            traj = self.ineq_corr(rad_points, cur_pos, self.lr, self.momentum, self.num_steps)

            return traj
        else:
            with torch.no_grad():
                scan_embedding = self.backbone(s[:,:360])  #(b,d)
                # scan_embedding = self.backbone_v0(rad_points.view(b,-1))  #(b,d)
                
                # # without recursive
                total_embedding = torch.cat([scan_embedding, s[:,-2:]], dim=1)
                if self.use_att:
                    traj_embedding = self.traj(total_embedding).view(b,-1,self.feats_dim)  #(b,l,d)
                else:
                    cur_pos = self.traj(total_embedding).view(b,self.length,2)  #(b,l,2)
                    for i in range(0,self.length):
                        if i == 0:
                            cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i])  #(b,l,2)
                        else:
                            cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i]) + cur_pos[:,i-1]

                # attention module
                if self.use_att:
                    # cur_pos = self.crossformer(cur_pos, s[:,:360])  #(b,l,d)
                    # cur_pos = self.crossformer(traj_embedding, s[:,:360])  #(b,l,d)
                    cur_pos = self.crossformer(traj_embedding, s[:,:360], s[:,-2:])  #(b,l,d)
                    cur_pos = self.traj_mlp(cur_pos)  #(b,l,2)
                    # cur_pos = 0.3 * torch.tanh(cur_pos)  #(b,l,2)

                # omega_data
                omega_data = None
                if self.use_att:
                    for i in range(0,self.length):
                        if i == 0:
                            linear_delta = self.linear_constraint * torch.tanh(cur_pos[:,i,0])  #(b,)
                            omega_delta  = self.omega_constraint  * torch.tanh(cur_pos[:,i,1])  #(b,)
                            cur_pos[:,i,0] = linear_delta * torch.cos(omega_delta)
                            cur_pos[:,i,1] = linear_delta * torch.sin(omega_delta)
                        else:
                            linear_delta = self.linear_constraint * torch.tanh(cur_pos[:,i,0])  #(b,)
                            omega_delta  = self.omega_constraint  * torch.tanh(cur_pos[:,i,1])  #(b,)
                            cur_pos[:,i,0] = linear_delta * torch.cos(omega_delta) + cur_pos[:,i-1,0]
                            cur_pos[:,i,1] = linear_delta * torch.sin(omega_delta) + cur_pos[:,i-1,1]

                traj = self.ineq_corr(rad_points, cur_pos, self.lr, self.momentum, self.num_steps)

                return traj