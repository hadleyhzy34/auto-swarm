import rospy
import os
import json
import numpy as np
import random
import time
import sys
import math
from collections import deque
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
from agent.network import Net
from env.env import Env
from torch.utils.tensorboard import SummaryWriter 

class Agent(nn.Module):
    def __init__(self, state_size, action_size, device = torch.device('cpu')):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000 # maximum steps per episode
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.device = device
        print(f'agent training device is loaded: {self.device}')
        
        # Buffer across all processes
        # self.memory = Buffer(self.memory_capacity, self.state_size, self.device)
        # self.share_memory = shared_buffer

        # DQN model
        # why using two q networks: https://ai.stackexchange.com/questions/22504/why-do-we-need-target-network-in-deep-q-learning
        self.lr = 1e-3
        self.tgt_net = Net(self.state_size, self.action_size, device = self.device)
        self.local_net = Net(self.state_size, self.action_size, device = self.device)
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=self.lr)

        # target network update frequency
        self.target_replace_iter = 100
        self.learn_step_counter = 0
        
        # loss
        self.loss = nn.MSELoss()

        # ros timer for training
        self.inference_freq = 10
        self.episode = 0  # current episode
        self.episodes = 6000 
        self.step = 0 # current steps
        self.step_episode = 500
        self.score = 0 # total reward for each episode
        # self.train_timer = rospy.Timer(rospy.Duration(1/self.inference_freq), self.collect_callback)

        # env
        # self.env = Env(self.action_size)
        self.distribution = torch.distributions.Normal
        self.action_scale = torch.tensor([0.11, 1.5]).unsqueeze(0) #(1,2)
        self.action_bias = torch.tensor([0.11, 0.]).unsqueeze(0) #(1,2)

    def choose_action(self, x):
        """
        Description:
        args:
            x: numpy, (state_size,)n
        return:
            
        """
        # import ipdb;ipdb.set_trace()
        x = torch.tensor(x,dtype=torch.float,device=self.device).unsqueeze(0) #(1, state_size)
        # if np.random.uniform() <= self.epsilon:
            # return np.random.randint(0, self.action_size)
        # else:
            # q_value = self.eval_net(x).detach()
            # action = q_value.max(1)[1].item()
        mean, sigma, _ = self.local_net(x)
        # print(f'mean: {mean}')
        assert mean.max() <= 1 and mean.min() >= -1, "action is out of range"
        mean = mean * self.action_scale + self.action_bias
        normal = self.distribution(mean, sigma)
        action = normal.rsample()
        action = torch.clamp(action, min=-self.action_scale + self.action_bias, max=self.action_scale + self.action_bias)
        # print(f'action:{action}')
        assert action[0,0] >= 0 and action[0,0] <= 0.22 and action[0,1] >= -1.5 and action[0,1] <= 1.5, "action is out of range"
        # action = torch.tanh(action)
        # action = action * self.action_scale + self.action_bias
        return action[0].detach().numpy()
    
    def learn(self, global_agent, done, next_state, buffer_s, buffer_a, buffer_r, shared_lock):
        """
        Description: calculate loss and update global net gradients and local net weights
        args:
        return:
        """
        if done:
            v_s_ = 0.               # terminal
        else:
            next_state = torch.tensor(next_state,dtype=torch.float,device=self.device).unsqueeze(0) #(1, state_size)
            _, _, v_s_ = self.local_net(next_state) #(batch_size, mu/sigma/value)

        buffer_v_target = []
        # print(f'buffer_r: {len(buffer_r)}')
        for r in buffer_r[::-1]:    # reverse buffer r
            v_s_ = r + self.gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        
        # print(f'buffer: {len(buffer_a)}')
        # calculate loss
        c_loss = 0
        a_loss = 0

        for (s,a,v_t) in zip(buffer_s, buffer_a, buffer_v_target):
            state = torch.tensor(s,dtype=torch.float,device=self.device).unsqueeze(0) #(1, state_size)
            mu,sigma,values = self.local_net(state)
            td = v_t - values
            c_loss += td.pow(2)
            
            m = self.distribution(mu,sigma)
            log_prob = m.log_prob(torch.from_numpy(a))
            # log_prob = torch.cumsum(log_prob, dim=1)[0,-1] #log_prob: log_prob(a0) + log_prob(a1) + ...
            # print(f'log_pro: {log_prob.shape}')
            entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
            # print(f'entropy: {entropy.shape}')
            exp_v = log_prob * td.detach() + 0.005 * entropy # gradient policy
            a_loss += -exp_v
        
        # print(f'c_loss: {c_loss.shape},{a_loss.shape}')
        loss = a_loss.mean() + c_loss[0,0]
        
        # calculate local gradients and push local parameters to global
        self.optimizer.zero_grad()
        loss.backward()

        shared_lock.acquire()
        try:
            global_agent.optimizer.zero_grad()
            for lp, gp in zip(self.local_net.parameters(), global_agent.local_net.parameters()):
                gp._grad = lp.grad
            global_agent.optimizer.step()
        finally:
            shared_lock.release()
        # pull global parameters
        self.local_net.load_state_dict(global_agent.local_net.state_dict())
