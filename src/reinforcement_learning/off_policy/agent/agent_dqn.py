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
from torch.utils.tensorboard import SummaryWriter

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Agent(nn.Module):
    def __init__(self, state_size,
                 action_size,
                 episode_step = 500,
                 tau = 0.05,
                 batch_size = 64,
                 memory_capacity = 5000,
                 epsilon = 0.9,
                 epsilon_decay = 0.95,
                 epsilon_min = 0.01,
                 lr = 1e-3,
                 device = torch.device('cpu')):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        # maximum steps per episode
        self.episode_step = episode_step
        self.tau = tau
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device
        print(f'agent training device is loaded: {self.device}')

        # Buffer
        self.memory_capacity = memory_capacity
        self.memory = Buffer(self.memory_capacity, self.state_size, self.device)

        # DQN model
        # why using two q networks: https://ai.stackexchange.com/questions/22504/why-do-we-need-target-network-in-deep-q-learning
        self.lr = lr
        self.tgt_net = DQN(self.state_size, self.action_size, self.device)
        self.eval_net = DQN(self.state_size, self.action_size, self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # target network update frequency
        # self.target_replace_iter = 100
        # self.learn_step_counter = 0

        # loss
        self.loss = nn.MSELoss()

        # ros timer for training
        self.inference_freq = 10
        self.episode = 0  # current episode
        self.episodes = 6000 
        self.step = 0 # current steps
        self.step_episode = 500
        self.score = 0 # total reward for each episode

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
        print(f'agent is updated')
        # update target network 
        # if self.learn_step_counter % self.target_replace_iter == 0:
        # self.tgt_net.load_state_dict(self.eval_net.state_dict())
        soft_update(self.tgt_net, self.eval_net, self.tau)
        # self.learn_step_counter += 1
