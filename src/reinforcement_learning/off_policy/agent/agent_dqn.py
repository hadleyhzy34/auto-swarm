#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

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
        
        # Buffer
        self.memory_capacity = 2000
        self.memory = Buffer(self.memory_capacity, self.state_size, self.device)
        
        # DQN model
        # why using two q networks: https://ai.stackexchange.com/questions/22504/why-do-we-need-target-network-in-deep-q-learning
        self.lr = 1e-3
        self.tgt_net = DQN(self.state_size, self.action_size, self.device)
        self.eval_net = DQN(self.state_size, self.action_size, self.device)
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
    
    def store_transition(self, state, action, reward, next_state, done):
        # import ipdb;ipdb.set_trace()
        state = torch.tensor(state, device = self.device)
        action = torch.tensor(action, device = self.device).unsqueeze(0) #(1,)
        reward = torch.tensor(reward, device = self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, device = self.device)
        done = torch.tensor(done, device = self.device).unsqueeze(0)
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        # import ipdb;ipdb.set_trace()
        # update target network 
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.tgt_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # buffer sampling
        mini_batch = self.memory.sample(self.batch_size)
        states = mini_batch[:,:self.state_size]
        next_states = mini_batch[:,self.state_size+3:]
        rewards = mini_batch[:,self.state_size+1]

        # actions to int
        actions = mini_batch[:,self.state_size].to(dtype=int)
        q_eval = self.eval_net(states).gather(1,actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.tgt_net(next_states).detach()
        q_target = rewards + self.gamma * q_next.max(1)[0]
        loss = self.loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    # def collect_callback(self, timer):
    #     if self.step == 0:
    #         self.state = self.env.reset()
        
    #     action = self.choose_action(self.state)
    #     next_state, reward, done = self.env.step(action)
    #     self.store_transition(self.state, action, reward, next_state, done)

    #     if self.memory.pointer >= self.memory.capacity:
    #         self.learn()
        
    #     self.score += reward
    #     self.state = next_state

    #     # update steps
    #     self.step += 1

    #     if self.step >= 500:
    #         rospy.loginfo("Time out!!!")
    #         done = True
        
    #     if done:
    #         print(f'Ep: {self.episode}, score: {self.score}, memory_capacity: {self.memory.pointer}, steps: {self.step}')
    #         self.score = 0
    #         self.episode += 1
    #         self.step = 0
            
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay