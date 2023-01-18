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
# from collections import deque
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
from agent.agent import Agent
from env.env import Env
from config.config import Config

# def train():
#     rospy.init_node('turtlebot3_dqn_stage')

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     state_size = 362
#     action_size = 5

#     agent = Agent(state_size, action_size, device)

#     rospy.spin()

def train(namespace, config):
    rospy.init_node(namespace)
    # initialize environment for each agent    
    env = Env(namespace, config)

    agent = Agent(config.Train.state_size, config.Train.action_size, config.Train.device)
    scores, episodes = [], []

    for e in range(config.Train.episodes):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)  # execute actions and wait until next scan(state)

            agent.store_transition(state, action, reward, next_state, done)

            if agent.memory.pointer >= agent.memory.capacity:
                agent.learn()

            score += reward
            state = next_state

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                print(f'Ep: {e}, score: {score}, memory_capacity: {agent.memory.pointer}, steps: {t}')
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

if __name__ == '__main__':
    train('tb3_0',Config)