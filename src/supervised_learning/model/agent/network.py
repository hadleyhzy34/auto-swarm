# from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np
from collections import namedtuple,deque
import torch.multiprocessing as mp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

################### buffer #############################
class Buffer(nn.Module):
    def __init__(self, capacity = 32, device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.capacity = capacity

        self.shared_reward = torch.zeros(capacity, device=self.device)
        self.shared_reward.share_memory_()
        
        self.pointer = torch.zeros(1,dtype=torch.int,device=self.device)
        self.pointer.share_memory_()

    def push_reward(self, score, shared_lock):
        shared_lock.acquire()
        try:
            self.shared_reward[self.pointer[0] % self.capacity] = score
            self.pointer[0] += 1
        finally:
            shared_lock.release()

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, scan_range = 360, device = torch.device('cpu')):
        super(Net, self).__init__()
        self.length = 10
        self.device = device
        
        self.backbone = nn.Sequential(
                nn.Linear(scan_range, 64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU()).to(device)

        self.plan = nn.Linear(32 + 6, 2)
        
        # self.action_mu = nn.Sequential(
        #         nn.Linear(32, action_dim),
        #         nn.Tanh()
        #         )

        # self.action_sigma = nn.Sequential(
        #         nn.Linear(32, action_dim),
        #         nn.Softplus())

        # self.value = nn.Sequential(
        #         nn.Linear(state_dim,128),
        #         nn.ReLU(),
        #         nn.Linear(128,1)
        #         )

        self.apply(weights_init_)
    
    def traj_generator(self, scan_embedding, pose, pos):
        """_summary_: generate next traj point

        Args:
            s (_type_): (b,32)
            pose: (b,2)
            pos: (b,2)
        
        return:
            next_pos: (b,2),(-1,1)
        """
        # print(f'shape: {scan_embedding.shape,pose.shape,pos.shape}')
        # print(f'shape: {scan_embedding.shape,pose,pos}')
        total_embedding = torch.cat([scan_embedding, pose, pos[None,:]], dim=1)
        total_embedding = self.plan(total_embedding)
        total_embedding = (torch.tanh(total_embedding) + torch.tensor([[1.,0.]])) * torch.tensor([[0.5,1]])

        # embedding phi
        # phi = 0.015 * (total_embedding[0,0] + 1) / 2
        # rou = total_embedding[0,1]
        return total_embedding

    def forward(self, s):
        """
        Description: actor policy
        args:
            s: (batch_size, state_dim: 360+2+2)
        return:
            a: (batch_size, action_dim)
        """
        # normalize lidar data and extract lidar data
        # print(f's:{s.device},model:{self.backbone}')
        scan_embedding = self.backbone(s[:,:360]/3.5)  #(b,32)
        
        cur_pos = torch.zeros((self.length,2),device=self.device)
        for i in range(self.length-1):
            cur_pos[i+1,:] = self.traj_generator(scan_embedding, s[:,360:], cur_pos[i,:])
        
        return cur_pos
