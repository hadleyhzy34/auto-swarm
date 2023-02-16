from turtle import pos

from numpy import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class CrossAttention(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, dropout=0., device=torch.device('cpu')):
        """
        description: cross attention module
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super(Attention,self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.device = device
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # radar data feature embedding
        self.conv_v = nn.Conv1d(in_channels=2, out_channels=self.inner_dim, kernel_size=3)
        self.conv_k = nn.Conv1d(in_channels=2, out_channels=self.inner_dim, kernel_size=3)
        
        # self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_k = nn.Linear(2, self.inner_dim, bias=False)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(2, self.inner_dim, bias=False)

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.eps = 1e-6
        self.use_goal = True

    def forward(self, traj_feats, data, goal):
        """
        Description:
        args:
            traj_feats: (b, l, 64), raw trajectory data
            data: (b, 360), raw radar data
            goal: (b, 2), goal position
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        b,l,d,h= *traj_feats.shape, self.heads

        # mask out lidar detection that are out of range
        lidar_mask = torch.lt(data,3.5)  #(b,360)
        # positional embedding
        pos_embedding = (torch.arange(0,360).to(self.device) * torch.pi / 180)[None,:,None].repeat(b,1,1)  #(b,360,1)
        data_feats = torch.cat([data[:,:,None],pos_embedding], dim=-1)  #(b,360,2)
        # goal positional embedding
        goal_feats = torch.cat([torch.linalg.norm(goal,dim=-1),torch.atan2(goal[:,1],goal[:,0])],dim=-1)[:,None,:]  #(b,1,2) 
        data_feats = torch.cat([data_feats,goal_feats],dim=1)  #(b,361,2)

        k_feats = self.conv_k(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 361, h*dim)
        v_feats = self.conv_v(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 361, h*dim)

        q_feats = self.to_q(traj_feats)  #(b,l,h*dim)

        q = rearrange(q_feats, 'b l (h d)-> b h l d', h=h)
        k = rearrange(k_feats, 'b m (h d)-> b h m d', h=h)
        v = rearrange(v_feats, 'b m (h d)-> b h m d', h=h)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  #(bhlm)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  #(bhld)
        out = rearrange(out, 'b h l d -> b l (h d)')  #(b,l,h*d)

        return self.to_out(out)  #(b,l,d)

class Attention(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, dropout=0., length=15, device=torch.device('cpu')):
        """
        description: cross attention module
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super(Attention,self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.device = device
        self.length = length

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(self.length*self.dim, self.length*self.inner_dim, bias=False)
        self.to_q = nn.Linear(self.length*self.dim, self.length*self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.length*self.dim, self.length*self.inner_dim, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.eps = 1e-6
        self.use_goal = True

    def forward(self, traj_feats):
        """
        Description:
        args:
            traj_feats: (b, l, d), traj embedding
        return:
            traj_feats: (b, l, d)
        """
        # pdb.set_trace()
        b,l,d,h= *traj_feats.shape, self.heads

        q_feats = self.to_q(traj_feats.view(b,-1)).view(b,l,-1)
        k_feats = self.to_k(traj_feats.view(b,-1)).view(b,l,-1)
        v_feats = self.to_v(traj_feats.view(b,-1)).view(b,l,-1)

        q = rearrange(q_feats, 'b l (h d)-> b h l d', h=h)
        k = rearrange(k_feats, 'b m (h d)-> b h m d', h=h)
        v = rearrange(v_feats, 'b m (h d)-> b h m d', h=h)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  #(bhlm)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  #(bhld)
        out = rearrange(out, 'b h l d -> b l (h d)')  #(b,l,h*d)

        return self.to_out(out)  #(b,l,d)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # pdb.set_trace()
        return self.fn(self.norm(x), **kwargs)

class CrossNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(361)
        self.fn = fn

    def forward(self, x, y):
        # pdb.set_trace()
        return self.fn(self.norm1(x), self.norm2(y))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, mlp_dim=64, dropout=0.,device=torch.device('cpu')):
        """
        description:
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.device = device

        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,device=self.device)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, traj_feats):
        """
        Description:
        args:
            traj_feats: (b, l, 64), traj embedding
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        for attn, ff in self.layers:
            # pdb.set_trace()
            x = attn(traj_feats) + traj_feats
            x = ff(x) + x
        return x

class Crossformer(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, mlp_dim=64, dropout=0.,device=torch.device('cpu')):
        """
        description:
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super().__init__()
        self.layers = nn.ModuleList([])

        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.device = device

        self.layers.append(nn.ModuleList([
            CrossNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,device=self.device)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, traj_feats, data):
        """
        Description:
        args:
            traj_feats: (b, l, 64), raw trajectory data
            data: (b, 360), raw radar data
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        for attn, ff in self.layers:
            # pdb.set_trace()
            x = attn(traj_feats, data) + traj_feats
            x = ff(x) + x
        return x