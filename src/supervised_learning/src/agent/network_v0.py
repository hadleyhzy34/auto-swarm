# from zmq import device
from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import numpy as np
from collections import namedtuple,deque
import torch.multiprocessing as mp
import pdb
from agent.attention_v0 import Crossformer,Transformer

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

        self.in_channels = 361 * 2
        self.out_channels = 64

        self.backbone = nn.Sequential(
            nn.Linear(self.in_channels,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,self.length*self.out_channels)
        ).to(device)
        
        self.traj_mlp = nn.Sequential(
            nn.Linear(self.feats_dim, 2)).to(device)

        self.use_att = True
        
        self.crossformer = Crossformer(self.feats_dim, self.heads, self.feats_dim, self.feats_dim, device=self.device)
        self.transformer = Transformer(self.feats_dim, self.heads, self.feats_dim, self.feats_dim, device=self.device)

    def forward(self, s):
        """
        Description: actor policy
        args:
            s: (batch_size, state_dim: 370)
        return:
            a: [length](batch_size, 2)
        """
        # pdb.set_trace()
        b = s.shape[0]

        pos_embedding = (torch.arange(0,360).to(self.device) * torch.pi / 180)[None,:,None].repeat(b,1,1)  #(b,360,1)
        input_embedding = torch.cat([s[:,:360][:,:,None],pos_embedding], dim=-1)  #(b,360,2)
        input_embedding = torch.cat([input_embedding,s[:,-2:][:,None,:]],dim=1) / 10  #(b,361,2)

        traj_embedding = self.backbone(input_embedding.view(b,-1)).view(b,self.length,self.feats_dim)  #(b,l,d)

        # attention module
        if self.use_att:
            # cur_pos = self.crossformer(traj_embedding, s[:,:360], s[:,-2:])  #(b,l,d)
            cur_pos = self.transformer(traj_embedding)  #(b,l,d)
            cur_pos = self.traj_mlp(cur_pos)  #(b,l,2)
            cur_pos = 0.3 * torch.tanh(cur_pos)  #(b,l,2)

        # pdb.set_trace()

        for i in range(1,self.length):
            cur_pos[:,i] += cur_pos[:,i-1]
        #     cur_pos[:,i+self.length] += cur_pos[:,i-1+self.length]

        return cur_pos
        # return cur_pos.view(-1,2,self.length).permute(0,2,1)
