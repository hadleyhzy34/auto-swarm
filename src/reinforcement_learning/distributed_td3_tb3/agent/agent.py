import rospy
import os
import json
import copy
import numpy as np
import random
import time
import sys
from collections import deque
from std_msgs.msg import Float32MultiArray
import pdb
import torch
import torch.nn as nn
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
        data = torch.tensor(data,dtype=torch.float,device=self.device)  #(batch_size,state_size,)
        # b = data.shape[0]
        
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
        
    def choose_action(self, x):
        """
        Description:
        args:
            x: numpy, (state_size,)n
        return:
            
        """
        # import ipdb;ipdb.set_trace()
        x = torch.tensor(x,dtype=torch.float,device=self.device).unsqueeze(0) #(1, state_size)
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q_value = self.eval_net(x).detach()
            action = q_value.max(1)[1].item()
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
        # import ipdb;ipdb.set_trace()
        # update target network 
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.tgt_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # buffer sampling
        mini_batch = shared_buffer.sample(self.batch_size).to(self.device)
        # mini_batch = self.memory.sample(self.batch_size)
        states = mini_batch[:,:self.state_size]
        next_states = mini_batch[:,self.state_size+3:]
        rewards = mini_batch[:,self.state_size+1]

        # actions to int
        actions = mini_batch[:,self.state_size].to(dtype=int)
        # print(f'action:{actions.device},states:{states.device},self.device:{self.device}')
        # print(f'res:{self.eval_net.to(self.device)(states).device}')

        #Note to use .to() method since eval and tgt net changed back to cpu method as actor
        q_eval = self.eval_net.to(self.device)(states).gather(1,actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.tgt_net.to(self.device)(next_states).detach()
        q_target = rewards + self.gamma * q_next.max(1)[0]
        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

if __name__ == '__main__':
    agent = Agent(370,2)
    lidar_data = np.random.uniform(low=0.,high=3.5,size=(1,360,))  #(1,360,)
    lidar_data[0,100:200] = 1.
    map = agent.preprocess(lidar_data)
    