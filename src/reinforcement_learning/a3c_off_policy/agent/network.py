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
        
        self.backbone = nn.Sequential(
                nn.Linear(scan_range, 64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU()).to(device)

        self.plan = nn.Sequential(nn.Linear(32 + (state_dim - scan_range), 32),nn.ReLU())
        
        self.action_mu = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Tanh()
                )

        self.action_sigma = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Softplus())

        self.value = nn.Sequential(
                nn.Linear(state_dim,128),
                nn.ReLU(),
                nn.Linear(128,1)
                )

        self.apply(weights_init_)

    def forward(self, s):
        """
        Description: actor policy
        args:
            s: (batch_size, state_dim)
        return:
            a: (batch_size, action_dim)
        """
        # normalize lidar data
        # print(f's:{s.device},model:{self.backbone}')
        scan_embedding = self.backbone(s[:,:360]/3.5)
        total_embedding = torch.cat([scan_embedding, s[:,360:]],dim=1)
        total_embedding = self.plan(total_embedding)
        # print(f'total_embedding shape is:{total_embedding.shape}')
        mu = self.action_mu(total_embedding)
        sigma = self.action_sigma(total_embedding)
        value = self.value(s)
        return mu, sigma, value
