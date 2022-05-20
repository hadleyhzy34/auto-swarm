import rospy
import os
import json
import numpy as np
import random
import time
import sys
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from concurrent.futures import process
from agent.agent_dqn import Agent
from env.env import Env
from config.config import Config


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
                # rospy.loginfo("Time out!!")
                env.status = 'Time out'
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                print(f"||namespace: {namespace}||Ep: {e:3d}||score: {score:3.2f}||goal_dist: {env.goal_distance:.2f}||memory_capacity: {agent.memory.pointer:3d}/2k||steps: {t:3d}||status: {env.status}")
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay


if __name__ == '__main__':
    # import ipdb;ipdb.set_trace()
    num_processes = Config.num_processes
    agent = Agent(Config.Train.state_size, Config.Train.action_size)
    agent.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(Config.namespace+str(rank),Config,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
