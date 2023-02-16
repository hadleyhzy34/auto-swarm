from turtle import pos

from numpy import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention_v0(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, dropout=0., device=torch.device('cpu')):
        """
        description: cross attention module
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super(Attention_v0,self).__init__()
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

        self.v_embd = nn.Linear(2*361,361*self.inner_dim)
        self.k_embd = nn.Linear(2*361,361*self.inner_dim)
        
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

    def forward(self, traj_feats, data, goal):
        """
        Description:
        args:
            traj_feats: (b, l, 64), raw trajectory data
            data: (b, 360), raw radar data
            goal: (b,2), goal position
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        b,l,d,h= *traj_feats.shape, self.heads

        # # positional embedding
        # pos_embedding = (torch.arange(0,360).to(self.device) * torch.pi / 180)[None,:,None].repeat(b,1,1)  #(b,360,1)
        # data_feats = torch.cat([data[:,:,None],pos_embedding], dim=-1)  #(b,360,2)
        # # data_feats = data + pos_embedding  #(b,360)

        pos_embedding = torch.arange(0,360).to(self.device) * torch.pi / 180  #(b,360)
        data_feats_x = data * torch.cos(pos_embedding)
        data_feats_y = data * torch.sin(pos_embedding)
        data_feats = torch.cat([data_feats_x[:,:,None], data_feats_y[:,:,None]],dim=-1)  #(b,360,2)
        data_feats = torch.cat([data_feats,goal[:,None,:]],dim=1)  #(b,361,2)
        # k_feats = self.to_k(data_feats)  #(b, 360, h*dim)
        # v_feats = self.to_v(data_feats)  #(b, 360, h*dim)

        k_feats = self.k_embd(data_feats.view(b,-1)).view(b,361,-1)
        v_feats = self.v_embd(data_feats.view(b,-1)).view(b,361,-1)

        # k_feats = self.conv_k(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 360, h*dim)
        # v_feats = self.conv_v(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 360, h*dim)

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

# class Attention(nn.Module):
#     def __init__(self, dim=64, heads=8, dim_head=64, dropout=0., device=torch.device('cpu')):
#         """
#         description: cross attention module
#         args:
#             dim: c_in||c_out
#             heads: number of heads, default: 8
#             dim_head: dim for each head
#         """
#         super(Attention,self).__init__()
#         self.dim = dim
#         self.inner_dim = dim_head * heads
#         self.dim_head = dim_head
#         self.device = device
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         # radar data feature embedding
#         self.conv_v = nn.Conv1d(in_channels=2, out_channels=self.inner_dim, kernel_size=3)
#         self.conv_k = nn.Conv1d(in_channels=2, out_channels=self.inner_dim, kernel_size=3)
        
#         # self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
#         self.to_k = nn.Linear(2, self.inner_dim, bias=False)
#         self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
#         self.to_v = nn.Linear(2, self.inner_dim, bias=False)

#         self.attend = nn.Softmax(dim = -1)

#         self.to_out = nn.Sequential(
#             nn.Linear(self.inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#         self.eps = 1e-6

#     def forward(self, traj_feats, data):
#         """
#         Description:
#         args:
#             traj_feats: (b, l, 64), raw trajectory data
#             data: (b, 360), raw radar data
#         return:
#             traj_feats: (b, l, 64)
#         """
#         # pdb.set_trace()
#         b,l,d,h= *traj_feats.shape, self.heads

#         # positional embedding
#         pos_embedding = (torch.arange(0,360).to(self.device) * torch.pi / 180)[None,:,None].repeat(b,1,1)  #(b,360,1)
#         data_feats = torch.cat([data[:,:,None],pos_embedding], dim=-1)  #(b,360,2)
#         # data_feats = data + pos_embedding  #(b,360)

#         # k_feats = self.to_k(data_feats)  #(b, 360, h*dim)
#         # v_feats = self.to_v(data_feats)  #(b, 360, h*dim)

#         k_feats = self.conv_k(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 360, h*dim)
#         v_feats = self.conv_v(data_feats.permute(0,2,1)).permute(0,2,1)  #(b, 360, h*dim)

#         q_feats = self.to_q(traj_feats)  #(b,l,h*dim)

#         q = rearrange(q_feats, 'b l (h d)-> b h l d', h=h)
#         k = rearrange(k_feats, 'b m (h d)-> b h m d', h=h)
#         v = rearrange(v_feats, 'b m (h d)-> b h m d', h=h)

#         dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  #(bhlm)

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)  #(bhld)
#         out = rearrange(out, 'b h l d -> b l (h d)')  #(b,l,h*d)

#         return self.to_out(out)  #(b,l,d)

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
        self.norm2 = nn.BatchNorm1d(360)
        self.norm3 = nn.BatchNorm1d(2)
        self.norm4 = nn.LayerNorm(362)
        self.fn = fn

    def forward(self, x, y, z):
        # pdb.set_trace()
        # return self.fn(self.norm1(x), self.norm2(y))
        # position normalization
        # y = y/10.
        # z = z/10.
        # return self.fn(self.norm1(x),y,z)
        # return self.fn(self.norm1(x), self.norm2(y), self.norm3(z))
        yz = self.norm4(torch.cat([y,z],dim=-1))
        return self.fn(self.norm1(x), yz[:,:360], yz[:,-2:])

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
            CrossNorm(dim, Attention_v0(dim, heads=heads, dim_head=dim_head, dropout=dropout,device=self.device)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, traj_feats, data, goal):
        """
        Description:
        args:
            traj_feats: (b, l, 64), raw trajectory data
            data: (b, 360), raw radar data
            goal:(b,2), goal position
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        for attn, ff in self.layers:
            # pdb.set_trace()
            x = attn(traj_feats, data, goal) + traj_feats
            x = ff(x) + x
        return x

class Attention_v1(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, dropout=0., device=torch.device('cpu')):
        """
        description: cross attention module
        args:
            dim: c_in||c_out
            heads: number of heads, default: 8
            dim_head: dim for each head
        """
        super(Attention_v1,self).__init__()
        self.dim = dim
        self.inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.device = device
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.inner_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.eps = 1e-6

    def forward(self, q, k, v):
        """
        Description:
        args:
            q: (b, l, hd), raw trajectory data
            k: (b, 361, hd)
            v: (b, 361, hd)
        return:
            traj_feats: (b, l, hd)
        """
        # pdb.set_trace()
        batch_size, l, _ = q.shape
        q = rearrange(q, 'b l (h d)-> b h l d', h=self.heads)  #(b,8,15,64)
        k = rearrange(k, 'b m (h d)-> b h m d', h=self.heads)  #(b,8,361,64)
        v = rearrange(v, 'b m (h d)-> b h m d', h=self.heads)  #(b,8,361,64)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  #(b h l 361), (b 8 15 361)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  #(bhld)
        out = rearrange(out, 'b h l d -> b l (h d)')  #(b,l,h*d)
        
        return out
        # return self.to_out(out)  #(b,l,d)


class Crossformer_v0(nn.Module):
    def __init__(self, dim=64, heads=8, dim_head=64, mlp_dim=64, length=15, dropout=0.,device=torch.device('cpu')):
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
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.device = device

        self.stages = 3
        self.att_layer = Attention_v1(dim, heads=heads, dim_head=dim_head, device=self.device)

        self.v_embd = nn.Linear(2 * 361, 361 * self.inner_dim)
        self.k_embd = nn.Linear(2 * 361, 361 * self.inner_dim)
        
        self.length = length
        self.to_q = nn.Linear(self.dim, self.inner_dim, bias=False)
        
        self.traj_embed = nn.Sequential(
            nn.Linear(self.length * 2, self.length * self.dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.length * self.dim * 2, self.length * self.dim),
            nn.LeakyReLU()
        )

        self.net = nn.Sequential(
            nn.Linear(self.inner_dim * 2, mlp_dim),
            nn.LayerNorm([self.length, mlp_dim]),
            # nn.InstanceNorm1d(mlp_dim),
            nn.LeakyReLU(),
            # nn.Linear(mlp_dim, 2),
            # nn.LeakyReLU()
            # nn.Tanh()
            # nn.GELU(),
            # nn.Linear(mlp_dim, mlp_dim),
        )

    def forward(self, traj, data, goal):
        """
        Description:
        args:
            traj_feats: (b, l, 2), raw trajectory data
            data: (b, 360, 2), raw radar data in 2d axis
            goal: (b, 2), goal position
        return:
            traj_feats: (b, l, 64)
        """
        # pdb.set_trace()
        b,l,_,h = *traj.shape, self.heads
        
        # q = self.to_q(traj_feats)  #(b,l,hd)
        data_feats = torch.cat([data,goal[:,None,:]],dim=1)  #(b,361,2)
        k = self.k_embd(data_feats.view(b,-1)).view(b,361,-1)  #(b,361,d)
        v = self.v_embd(data_feats.view(b,-1)).view(b,361,-1)  #(b,361,d)

        traj_feats = self.traj_embed(traj.view(b,-1)).view(b,self.length,-1)  #(b.l.d)
        # q = self.to_q(traj.view(b,-1)).view(b,self.length,-1)  #(b,l,hd)
        
        # pdb.set_trace()
        for _ in range(self.stages):
            # q = self.to_q(traj.view(b,-1)).view(b,self.length,-1)  #(b,l,hd)
            q = self.to_q(traj_feats)  #(b,l,hd)
            qkv = self.att_layer(q, k, v)
            # traj = traj + self.net(torch.cat([q,qkv],dim=-1))
            # traj_feats = q + self.net(torch.cat([q, qkv], dim = -1))
            # traj_feats = q + self.net(qkv)  #(b,l,d)
            traj_feats = traj_feats + self.net(torch.cat([q,qkv],dim=-1))  #(b,l,d)
        
        return traj_feats
