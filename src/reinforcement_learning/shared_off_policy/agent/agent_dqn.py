import rospy
import os
import json
import numpy as np
import random
import time
import sys
from collections import deque
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
from agent.network import DQN,Buffer
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
        self.tgt_net = DQN(self.state_size, self.action_size, device = self.device)
        self.eval_net = DQN(self.state_size, self.action_size, device = self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

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
