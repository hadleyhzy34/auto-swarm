import rospy
import os
import json
import numpy as np
import random
import time
import sys
import pdb
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

        # loss
        self.loss = nn.MSELoss()
        self.cql = True

    def cql_loss(self, q_values, current_action):
        """
        Description: Computes the CQL loss for a batch of Q-values and actions.
        """
        # pdb.set_trace()
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action[None,:])

        return (logsumexp - q_a).mean()

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
        q_evals = self.eval_net(states)
        q_eval = q_evals.gather(1,actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_next = self.tgt_net(next_states).detach()
            q_target = rewards + self.gamma * q_next.max(1)[0]

        loss = self.loss(q_eval, q_target)
        if self.cql:
            loss += self.cql_loss(q_evals, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network 
        soft_update(self.tgt_net, self.eval_net, self.tau)
