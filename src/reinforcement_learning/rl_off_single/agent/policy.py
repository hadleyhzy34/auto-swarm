import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
import pdb
import torchvision.models as models
from agent.buffer import ReplayBuffer

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

class QNetwork(nn.Module):
    def __init__(self, backbone_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        
        self.q1 = nn.Sequential(
                nn.Linear(backbone_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                )

        self.q2 = nn.Sequential(
                nn.Linear(backbone_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                )
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        # pdb.set_trace()
        x_state_action = torch.cat([state, action], 1)
        
        x1 = self.q1(x_state_action)
        x2 = self.q2(x_state_action)
        
        return x1, x2

class PolicyNetwork(nn.Module):
    def __init__(self, backbone_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.model = nn.Sequential(
                nn.Linear(backbone_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                )
        
        # self.linear1 = nn.Linear(state_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        state = self.model(state)  #(b,h)
        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mean, log_std  #(b,a), (b,a)

    def sample(self, state, epsilon=1e-6):
        # pdb.set_trace()
        mean, log_std = self.forward(state)
        # print(f'mean shape:{mean.shape}, log_std shape: {log_std.shape}')
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

class Agent(object):
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, 
                 tau=1e-2, 
                 alpha=0.2, 
                 hidden_dim=1024,
                 backbone_dim=96,
                 replay_buffer_size=50000,
                 batch_size=32,
                 episode_step=500,
                 lr=0.001,
                 update_step=500,
                 device=torch.device('cpu')):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.episode_step = episode_step # maximum steps per episode
        self.epsilon = 0.01
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.batch_size = batch_size

        # self.action_range = [action_space.low, action_space.high]
        self.lr=lr
        self.hidden_dim = hidden_dim
        self.backbone_dim = backbone_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.update_step = update_step

        self.target_update_interval = 1
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.device = device

        #(h,w,d)->(576)
        self.backbone = list(models.mobilenet_v3_small(weights=None).to(self.device).children())[0]
        self.backbone = torch.nn.Sequential(*(self.backbone[:-1]))
        # self.backbone = torch.nn.Sequential(*(list(models.mobilenet_v3_small(weights=None).to(self.device).children())[:-1]))
        # models.mobilenet_v3_small(weights=None).to(self.device)
        # pdb.set_trace()
        self.critic = QNetwork(backbone_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.critic_target = QNetwork(backbone_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        print('entropy', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
    
        self.policy = PolicyNetwork(backbone_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        # agent action parameters
        self.max_ang_vel = 1.5
        self.max_lin_vel = 0.15
        self.update = 0

    def select_action(self, state, eval=False):
        # pdb.set_trace()
        state = state.to(device=self.device).unsqueeze(0).permute(0,3,1,2)  #(1,3,360,256)
        state = self.backbone(state)[:,:,0,0]  #(1,backbone_dim,)

        if np.random.uniform() <= self.epsilon:
            action = torch.rand(1,self.action_dim) * 2 - 1
            # print('random action is selected')
        else:
            # print('reasonable action is selected')
            if eval is False:
                action, _, _, _ = self.policy.sample(state) #action in range of [-1,1]
            else:
                _, _, action, _ = self.policy.sample(state)
                action = torch.tanh(action)

        # action range based on robot physics
        action = action.detach().cpu().numpy()[0]
        action = (action + np.array([1.,0.])) * np.array([0.5 * self.max_lin_vel,self.max_ang_vel])

        # test case
        if action[0] < 0 or action[0] > self.max_lin_vel:
            raise Exception("linear velocity is out of range")
        if action[1] < -self.max_ang_vel or action[1] > self.max_ang_vel:
            raise Exception("angular velocity is out of range")

        # print(f'linear: {action[0]}, angular: {action[1]}')
        return action

    def learn(self, memory, batch_size):
        # pdb.set_trace()
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)  #(b,l)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # pdb.set_trace()
        with torch.no_grad():
            #vf_next_target = self.value_target(next_state_batch)
            #next_q_value = reward_batch + (1 - done_batch) * self.gamma * (vf_next_target)
            next_state_feat_batch = self.backbone(next_state_batch.permute(0,3,1,2))[:,:,0,0]  #(b,backbone_dim)
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_feat_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_feat_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)
        
        state_feat_batch = self.backbone(state_batch.permute(0,3,1,2))[:,:,0,0]  #(b,backbone_dim)
        qf1, qf2 = self.critic(state_feat_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # 
        qf2_loss = F.mse_loss(qf2, next_q_value) # 
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # for training stability, detach backbone when backward actor Loss
        pi, log_pi, mean, log_std = self.policy.sample(state_feat_batch.detach())

        qf1_pi, qf2_pi = self.critic(state_feat_batch.detach(), pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 
        # Regularization Loss
        #reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        #policy_loss += reg_loss

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        #alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        if self.update % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            self.update += 1

        # print(f'system parameters are updated')
        #return vf_loss.item(), qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    # Save model parameters
    def save_models(self, episode_count):
        torch.save(self.policy.state_dict(),dirPath+'/model/'+str(episode_count)+ '_policy_net.pth')
        torch.save(self.critic.state_dict(), dirPath  +  '/model/'+str(episode_count)+ 'value_net.pth')
        # hard_update(self.critic_target, self.critic)
        # torch.save(soft_q_net.state_dict(), dirPath + '/SAC_model/' + world + '/' + str(episode_count)+ 'soft_q_net.pth')
        # torch.save(target_value_net.state_dict(), dirPath + '/SAC_model/' + world + '/' + str(episode_count)+ 'target_value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")
    
    # Load model parameters
    def load_models(self, episode):
        self.policy.load_state_dict(torch.load(dirPath + '/model/' +str(episode)+ '_policy_net.pth'))
        self.critic.load_state_dict(torch.load(dirPath + '/model/' +str(episode)+ 'value_net.pth'))
        hard_update(self.critic_target, self.critic)
        # soft_q_net.load_state_dict(torch.load(dirPath + '/SAC_model/' + world + '/'+str(episode)+ 'soft_q_net.pth'))
        # target_value_net.load_state_dict(torch.load(dirPath + '/SAC_model/' + world + '/'+str(episode)+ 'target_value_net.pth'))
        print('***Models load***')
