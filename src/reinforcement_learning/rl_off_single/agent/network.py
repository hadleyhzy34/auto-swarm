import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchvision.models as models
import random
import numpy as np
from collections import namedtuple,deque
import pdb

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

################### buffer #############################
class Buffer:
    def __init__(self, capacity, state_dim, device = torch.device('cpu')):
        self.capacity = capacity
        self.pointer = 0
        self.device = device
        self.memory = torch.empty((self.capacity, state_dim + 1 + 1 + state_dim + 1),device=self.device)
    
    def push(self, state, action, reward, next_state, done):
        """
        Save a transition
        Args:
            state, action, reward, next_state, done
        Func: store transition status into replay buffer
        """
        # import ipdb;ipdb.set_trace()
        self.memory[self.pointer % self.capacity] = torch.cat([state,action,reward,next_state,done],0)
        self.pointer += 1

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

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.model = nn.Sequential(
                nn.Linear(hidden_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024,64),
                nn.ReLU(),
                nn.Linear(64,2),
                nn.Tanh()
                )

    def forward(self, state):
        x = self.model(state)
        return x



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, device = torch.device('cpu')):
        super(DQN, self).__init__()

        # self.backbone = models.mobilenet_v3_small(weights=None)
        # self.dropout = 0.2
        self.model = nn.Sequential(
                nn.Linear(576, 1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024,64),
                nn.ReLU(),
                nn.Linear(64,2),
                nn.Tanh()
                ).to(device)

        self.apply(weights_init_)

        self.backbone = list(models.mobilenet_v3_small(weights=None).to(self.device).children())[0]
        self.backbone = torch.nn.Sequential(*(self.backbone[:-1]))

    def forward(self, s):
        '''
        description: policy network
        args:
            s: (c,h,w)
        '''
        pdb.set_trace()
        hidden_s = self.backbone(s)
        return self.model(hidden_s)  #(batch, action_dim)
