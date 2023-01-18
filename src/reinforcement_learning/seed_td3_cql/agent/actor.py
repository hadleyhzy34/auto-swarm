import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pdb

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        patch_height = 16
        patch_width = 16
        self.heads = 12
        patch_dim = patch_height * patch_width * 1
        self.dim = patch_height * patch_width * 1
        self.num_patches = (224//patch_height) * (224//patch_width)
        self.scale = state_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)
        
        # state embedding
        self.state_dim = 64
        self.state_embedding = nn.Linear(state_dim - 360, self.state_dim)
        self.to_q = nn.Linear(self.state_dim, self.state_dim * self.heads, bias=False)
        
        print(f'check actor init here')
        # image to path
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, self.dim),
        )
        self.to_k = nn.Linear(self.dim, self.state_dim * self.heads, bias=False)
        self.to_v = nn.Linear(self.dim, self.state_dim * self.heads, bias=False)
        
        self.ffn = nn.Linear(self.state_dim, self.state_dim)
        self.to_mlp = nn.Linear(self.state_dim * self.heads, self.state_dim)
        self.to_out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LeakyReLU(),
            nn.Linear(self.state_dim, self.action_dim)
        )
        print(f'check actor end here')
    
    def forward(self, grid, state):
        """description: actor policy
        Args:
            grid (b,224,224): lidar grid map
            state (b,4): yaw, omega, goal_pos.x, goal_pos.y
        """
        # pdb.set_trace()
        # patch embedding
        x = self.to_patch_embedding(grid[:,None,:,:])  #(b,h*w,dim)
        b,n,_ = x.shape
        
        # state embedding
        state_emd = self.state_embedding(state)  #(b,state_dim)
        q_feats = self.to_q(state_emd).view(b,1,-1)  #(b,1,h*d)
        
        # k,v embedding
        k_feats = self.to_k(x.view(-1, self.dim)).view(b,self.num_patches,-1)  #(b,l,h*d)
        v_feats = self.to_v(x.view(-1, self.dim)).view(b,self.num_patches,-1)  #(b,l,h*d)

        q = rearrange(q_feats, 'b l (h d)-> b h l d', h=self.heads)
        k = rearrange(k_feats, 'b m (h d)-> b h m d', h=self.heads)
        v = rearrange(v_feats, 'b m (h d)-> b h m d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1,-2)) * self.scale  #(b,h,1,l)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)  #(bhld)
        out = rearrange(out, 'b h l d -> b l (h d)')  #(b,l,h*d)
        out = self.to_mlp(out)  #(b,1,d)
        
        action_embd = state_emd[:,None,:] + out
        action_embd = action_embd + self.ffn(action_embd)  #(b,1,state_dim)

        return self.to_out(action_embd)[:,0,:]  #(b,a)

if __name__ == '__main__':
    state = torch.randn(2,4)
    img = torch.randn(2,224,224)
    act = Actor(364,2)
    res = act(img,state)