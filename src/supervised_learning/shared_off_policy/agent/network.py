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
    def __init__(self, capacity, state_dim, device = torch.device('cpu')):
        super().__init__()
        self.capacity = capacity
        self.device = device
        self.memory = torch.empty((self.capacity, state_dim + 1 + 1 + state_dim + 1),device=self.device)
        # share this memory across multiprocessing
        self.memory.share_memory_()
        # share pointer across multi processes
        self.pointer = torch.zeros(1,dtype=torch.int,device=self.device)
        self.pointer.share_memory_()

        # shared reward for training visualization
        self.reward_capacity = 100
        self.reward_pointer = torch.zeros(1, dtype=torch.int, device=self.device)
        self.reward_pointer.share_memory_()

        self.shared_reward = torch.zeros(self.reward_capacity, device=self.device)
        self.shared_reward.share_memory_()
    
    def push_reward(self, score, shared_lock):
        shared_lock.acquire()
        try:
            self.shared_reward[self.reward_pointer[0] % self.reward_capacity] = score
            self.reward_pointer[0] += 1
        finally:
            shared_lock.release()
    

    def push(self, state, action, reward, next_state, done, shared_lock):
        """
        Save a transition
        Args:
            state, action, reward, next_state, done
        Func: store transition status into replay buffer
        """
        shared_lock.acquire()
        try:
            # import ipdb;ipdb.set_trace()
            # print(f'pointer: {self.pointer[0] % self.capacity},{(self.pointer[0] % self.capacity).dtype}')
            self.memory[self.pointer[0] % self.capacity] = torch.cat([state,action,reward,next_state,done],0)
            self.pointer[0] += 1
            # print(f'current pointer is: {self.pointer}')
        finally:
            shared_lock.release()
    
    def sample(self, batch_size):
        '''
        randomly select batch_size samples for network update
        Args:
            batch_size
        Return:
            state: tuple(batch_size,)
            action:tuple(batch_size,)
            reward:tuple(batch_size,)
            next_state:tuple(batch_size,)
        '''
        return self.memory[torch.randint(self.capacity, (batch_size,)),:]
 
    def __len__(self)->int:
        '''
        return length of buffer list
        '''
        return self.pointer % self.capacity

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, scan_range = 360, device = torch.device('cpu')):
        super(DQN, self).__init__()
        
        self.backbone = nn.Sequential(
                nn.Linear(scan_range, 64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU()).to(device)

        # self.dropout = 0.2
        self.model = nn.Sequential(
                nn.Linear(32 + (state_dim - scan_range), 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
                ).to(device)

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
        # print(f'total_embedding shape is:{total_embedding.shape}')
        return self.model(total_embedding)
