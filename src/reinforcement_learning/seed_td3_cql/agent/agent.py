# import rospy
import os
import json
import copy
import numpy as np
import random
import time
import sys
from collections import deque
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from actor import Actor
from critic import Q_Critic
# from agent.network import DQN,Buffer
# from env.env import Env
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
        self.epsilon_min = 0.01
        self.batch_size = 256
        self.device = device
        print(f'agent training device is loaded: {self.device}')
        
        # Buffer across all processes
        # self.memory = Buffer(self.memory_capacity, self.state_size, self.device)
        # self.share_memory = shared_buffer

        self.a_lr = 1e-4
        self.c_lr = 1e-4
        
        # td3 policy and critic network
        self.actor = Actor(self.state_size,self.action_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.a_lr)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.q_critic = Q_Critic(self.state_size, self.action_size).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr = self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

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

    def preprocess(self, data):
        """"
        description: lidar data to grid image
        args:
            data: (state_size,)
        return:
            map: (224,224), torch.floatTensor
        """
        # pdb.set_trace()
        data = torch.tensor(data,dtype=torch.float,device=self.device)[0]  #(batch_size,scan_size,)
        
        # pdb.set_trace()
        rad_points = torch.zeros((360, 2),device=self.device)  #(360,2)
        rad_points[:,0] = torch.cos((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[0:360]
        rad_points[:,1] = torch.sin((torch.arange(0,360).to(self.device)) * torch.pi / 180) * data[0:360]
        
        # plt.figure()
        # plt.scatter(rad_points[:,0].cpu().numpy(),rad_points[:,1].cpu().numpy())
        # # plt.show()
        # plt.savefig('test1.png')
        
        #voxelize 2d lidar points
        rad_points[:,0] -= -3.5
        rad_points[:,1] = 3.5 - rad_points[:,1]
        rad_points = rad_points.div((3.5*2)/224,rounding_mode='floor').long()
         
        img = torch.zeros((224,224),device = self.device)  #(224,224)
        img[rad_points[:,0],rad_points[:,1]] = 1.
        
        # remove center point
        img[112,112] = 0.
        
        # plt.figure()
        # plt.imshow(img.numpy())
        # # plt.show()
        # plt.savefig('test2.png')
        
        return img
        
    def choose_action(self, state):
        """
        Description:
        args:
            state: numpy, (b,state_size,)n
        return:
            
        """
        # import ipdb;ipdb.set_trace()
        img = self.preprocess(state[:,:360])  #(224,224)
        # pdb.set_trace()
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = state[:,-4:]  #(b,4)
        action = self.actor(img[None,:], state)[0]  #(2,)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done, shared_buffer, shared_lock):
        # import ipdb;ipdb.set_trace()
        state = torch.tensor(state, device = self.device)
        action = torch.tensor(action, device = self.device).unsqueeze(0) #(1,)
        reward = torch.tensor(reward, device = self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, device = self.device)
        done = torch.tensor(done, device = self.device).unsqueeze(0)
        
        # print(f'state:{state.shape,state.dtype}, action:{action.shape,action.dtype},reward:{reward.shape,reward.dtype},next_state:{next_state.shape,next_state.dtype},done:{done.shape,done.dtype}')
        shared_buffer.push(state, action, reward, next_state, done, shared_lock)
    
    def learn(self, shared_buffer):
        # buffer sampling
        mini_batch = shared_buffer.sample(self.batch_size).to(self.device)
        # mini_batch = self.memory.sample(self.batch_size)
        states = mini_batch[:,:self.state_size]
        next_states = mini_batch[:,self.state_size+2:-1]
        rewards = mini_batch[:,self.state_size+1]
        dones = mini_batch[:,-1]

        # print(f'mini_batch shape: {mini_batch.shape}, next_states: {next_states.shape}')
        # actions to int
        actions = mini_batch[:,self.state_size].to(dtype=int)
        
        self.delay_counter += 1
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = (
					self.actor_target(next_states) + noise  # Noisy on target action
			).clamp(-self.max_action, self.max_action)
        
        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(next_states, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)

        target_Q = rewards + self.gamma * target_Q * (1 - dones)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(states, actions)
        
        # compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        
        if self.delay_counter == self.delay_freq:
			# Update Actor
            a_loss = - self.q_critic.Q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()
        
            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            self.delay_counter = -1

if __name__ == '__main__':
    agent = Agent(364,2)
    lidar_data = np.random.uniform(low=0.,high=3.5,size=(1,364,))  #(1,360,)
    lidar_data[0,100:200] = 1.
    test = agent.choose_action(lidar_data)    