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

        # self.backbone = nn.Sequential(
        #         nn.Linear(scan_range, 360),
        #         nn.ReLU(),
        #         nn.Linear(360, 360),
        #         nn.ReLU(),
        #         nn.Linear(360, self.feats_dim),
        #         nn.ReLU()).to(device)
        
        self.backbone = nn.Sequential(
                nn.Linear(scan_range, 360),
                # nn.BatchNorm1d(360),
                nn.ReLU(),
                # nn.Linear(360, 360),
                # nn.ReLU(),
                nn.Linear(360, self.feats_dim),
                # nn.BatchNorm1d(self.feats_dim),
                nn.ReLU()).to(device)
        
        # self.plan = nn.Sequential(
        #         nn.Linear(32 + 4, 64),
        #         nn.ReLU(),
        #         nn.Linear(64,  2)).to(device)
        
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

        # self.apply(weights_init_)
    
    def traj_generator(self, scan_embedding, tgt_pose, cur_pose):
        """_summary_: generate next traj point

        Args:
            s (_type_): (b,64)
            tgt_pose: (b,2)
            cur_pose: (b,2)
        return:
            next_pos: (b,2),(-0.3,0.3)
        """
        # total_embedding = torch.cat([scan_embedding, self.norm1(tgt_pose), self.norm1(cur_pose)], dim=1)
        total_embedding = torch.cat([scan_embedding, tgt_pose, cur_pose], dim=1)
        # pos = self.traj_decoder(total_embedding)  #(b,d)
        total_embedding = self.plan(total_embedding)  #(b,d)

        pos = self.plan_decoder(total_embedding)
        pos = 0.3 * torch.tanh(pos).to(self.device)

        return total_embedding, pos

    def forward(self, s, mode='train'):
        """
        Description: actor policy
        args:
            s: (batch_size, state_dim: 360+2+2+2)
        return:
            a: [length](batch_size, 2)
        """
        if mode == 'train':
            # pdb.set_trace()
            b = s.shape[0]
            # normalize lidar data and extract lidar data
            # print(f's:{s.device},model:{self.backbone}')
            # scan_embedding = self.backbone(s[:,:360]/3.5)  #(b,32)
            # clamp inf of data that are out of range
            # s[:,:360] = torch.clamp(s[:,:360],max=3.5)

            # rad_points = torch.zeros((self.batch_size, 360, 2),device=self.device)
            # rad_points[:,:,0] = torch.cos((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]
            # rad_points[:,:,1] = torch.sin((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]

            # scan_embedding = self.backbone(s[:,:360]/3.5 - 0.5)  #(b,32)
            # scan_embedding = self.backbone(s[:,:360]/3.5)  #(b,32),(0,1)
            # scan_embedding = self.backbone(s[:,:360]/10.)  #(b,32),(0,1)
            scan_embedding = self.backbone(s[:,:360])  #(b,32),(0,1)

            # cur_pos = torch.zeros((b,self.length,2),device=self.device)
            # cur_pos = []
            # cur_pos.append(torch.zeros(b,2,device=self.device))
            # for i in range(self.length-1):
            #     cur_pos.append(self.traj_generator(scan_embedding, s[:,-2:], cur_pos[-1]) + cur_pos[-1])
            
            # # without recursive
            # pdb.set_trace()
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

            # # # with recursive
            # # cur_pos = torch.zeros((b,1,2),device=self.device)  #relative positions
            # # pre_pos = cur_pos[:,-1,:]  #(b,2)
            # # traj_embedding = None
            # for i in range(self.length):
            #     if i == 0:
            #         # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:]/10., torch.zeros((b,2),device=self.device))
            #         cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], torch.zeros((b,2),device=self.device))
            #         traj_embedding = cur_embedding[:,None,:]
            #         # cur_pos = 0.3 * torch.tanh(pos)[:,None,:]
            #         cur_pos = pos[:,None,:]  #(b,1,2)
            #     else:
            #         # # pdb.set_trace()
            #         # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:])
            #         # traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
            #         # # pos = 0.3 * torch.tanh(pos) + cur_pos[:,-1,:]
            #         # pos += cur_pos[:,-1,:]
            #         # cur_pos = torch.cat([cur_pos, pos[:,None,:]],dim=1)  #(b,i+1,2)
                    
            #         # detach previous position
            #         # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:]/10., cur_pos[:,-1,:].detach()/10.)
            #         cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:].detach())
            #         traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
            #         # pos = 0.3 * torch.tanh(pos) + cur_pos[:,-1,:]
            #         # pos += cur_pos[:,-1,:].detach()
            #         pos += cur_pos[:,-1,:]
            #         cur_pos = torch.cat([cur_pos, pos[:,None,:]],dim=1)  #(b,i+1,2)
            
            # pdb.set_trace()

                # # pdb.set_trace()
                # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:]+pre_pos)
                # if traj_embedding == None:
                #     traj_embedding = cur_embedding[:,None,:]
                # else:
                #     traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
                # cur_pos = torch.cat([cur_pos, pos[:,None,:] + pos],dim=1)  #(b,i+1,2)
                # pre_pos += cur_pos[:,-1,:]  #(b,1,2)

            # attention module
            if self.use_att:
                # cur_pos = self.crossformer(cur_pos, s[:,:360])  #(b,l,d)
                # cur_pos = self.crossformer(traj_embedding, s[:,:360])  #(b,l,d)
                cur_pos = self.crossformer(traj_embedding, s[:,:360], s[:,-2:])  #(b,l,d)
                cur_pos = self.traj_mlp(cur_pos)  #(b,l,2)
                # cur_pos = 0.3 * torch.tanh(cur_pos)  #(b,l,2)

            if self.use_att:
                for i in range(0,self.length):
                    if i == 0:
                        cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i])  #(b,l,2)
                    else:
                        cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i]) + cur_pos[:,i-1]
                #     cur_pos[:,i+self.length] += cur_pos[:,i-1+self.length]
            # pdb.set_trace()
        else:
            with torch.no_grad():
                # pdb.set_trace()
                b = s.shape[0]
                # normalize lidar data and extract lidar data
                # print(f's:{s.device},model:{self.backbone}')
                # scan_embedding = self.backbone(s[:,:360]/3.5)  #(b,32)
                # clamp inf of data that are out of range
                # s[:,:360] = torch.clamp(s[:,:360],max=3.5)

                # rad_points = torch.zeros((self.batch_size, 360, 2),device=self.device)
                # rad_points[:,:,0] = torch.cos((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]
                # rad_points[:,:,1] = torch.sin((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[:,0:360]

                # scan_embedding = self.backbone(s[:,:360]/3.5 - 0.5)  #(b,32)
                # scan_embedding = self.backbone(s[:,:360]/3.5)  #(b,32),(0,1)
                # scan_embedding = self.backbone(s[:,:360]/10.)  #(b,32),(0,1)
                scan_embedding = self.backbone(s[:,:360])  #(b,32),(0,1)

                # cur_pos = torch.zeros((b,self.length,2),device=self.device)
                # cur_pos = []
                # cur_pos.append(torch.zeros(b,2,device=self.device))
                # for i in range(self.length-1):
                #     cur_pos.append(self.traj_generator(scan_embedding, s[:,-2:], cur_pos[-1]) + cur_pos[-1])
                
                # # # without recursive
                # # pdb.set_trace()
                # total_embedding = torch.cat([scan_embedding, s[:,-2:]], dim=1)
                # if self.use_att:
                #     traj_embedding = self.traj(total_embedding).view(b,-1,self.feats_dim)  #(b,l,d)
                # else:
                #     cur_pos = self.traj(total_embedding).view(b,self.length,2)  #(b,l,2)
                #     for i in range(0,self.length):
                #         if i == 0:
                #             cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i])  #(b,l,2)
                #         else:
                #             cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i]) + cur_pos[:,i-1]

                # # with recursive
                # cur_pos = torch.zeros((b,1,2),device=self.device)  #relative positions
                # pre_pos = cur_pos[:,-1,:]  #(b,2)
                # traj_embedding = None
                for i in range(self.length):
                    if i == 0:
                        # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:]/10., torch.zeros((b,2),device=self.device))
                        cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], torch.zeros((b,2),device=self.device))
                        traj_embedding = cur_embedding[:,None,:]
                        # cur_pos = 0.3 * torch.tanh(pos)[:,None,:]
                        cur_pos = pos[:,None,:]  #(b,1,2)
                    else:
                        # # pdb.set_trace()
                        # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:])
                        # traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
                        # # pos = 0.3 * torch.tanh(pos) + cur_pos[:,-1,:]
                        # pos += cur_pos[:,-1,:]
                        # cur_pos = torch.cat([cur_pos, pos[:,None,:]],dim=1)  #(b,i+1,2)
                        
                        # detach previous position
                        # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:]/10., cur_pos[:,-1,:].detach()/10.)
                        cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:].detach())
                        traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
                        # pos = 0.3 * torch.tanh(pos) + cur_pos[:,-1,:]
                        # pos += cur_pos[:,-1,:].detach()
                        pos += cur_pos[:,-1,:]
                        cur_pos = torch.cat([cur_pos, pos[:,None,:]],dim=1)  #(b,i+1,2)
                
                # pdb.set_trace()

                    # # pdb.set_trace()
                    # cur_embedding, pos = self.traj_generator(scan_embedding, s[:,-2:], cur_pos[:,-1,:]+pre_pos)
                    # if traj_embedding == None:
                    #     traj_embedding = cur_embedding[:,None,:]
                    # else:
                    #     traj_embedding = torch.cat([traj_embedding, cur_embedding[:,None,:]], dim=1)  #(b,i,d)
                    # cur_pos = torch.cat([cur_pos, pos[:,None,:] + pos],dim=1)  #(b,i+1,2)
                    # pre_pos += cur_pos[:,-1,:]  #(b,1,2)

                # attention module
                if self.use_att:
                    # cur_pos = self.crossformer(cur_pos, s[:,:360])  #(b,l,d)
                    # cur_pos = self.crossformer(traj_embedding, s[:,:360])  #(b,l,d)
                    cur_pos = self.crossformer(traj_embedding, s[:,:360], s[:,-2:])  #(b,l,d)
                    cur_pos = self.traj_mlp(cur_pos)  #(b,l,2)
                    # cur_pos = 0.3 * torch.tanh(cur_pos)  #(b,l,2)

                if self.use_att:
                    for i in range(0,self.length):
                        if i == 0:
                            cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i])  #(b,l,2)
                        else:
                            cur_pos[:,i] = 0.3 * torch.tanh(cur_pos[:,i]) + cur_pos[:,i-1]
                    #     cur_pos[:,i+self.length] += cur_pos[:,i-1+self.length]
                # pdb.set_trace()
        return cur_pos
        # return cur_pos.view(-1,2,self.length).permute(0,2,1)