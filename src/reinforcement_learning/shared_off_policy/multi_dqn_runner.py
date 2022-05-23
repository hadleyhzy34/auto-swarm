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
from multiprocessing import Lock
from concurrent.futures import process
from agent.agent_dqn import Agent
from agent.network import Buffer
from env.env import Env
from config.config import Config


def train(namespace, config, gAgent, shared_buffer, shared_lock):
    rospy.init_node(namespace)
    # initialize environment for each agent    
    env = Env(namespace, config)

    agent = Agent(config.Train.state_dim, config.Train.action_dim, config.Train.device)
    scores, episodes = [], []

    for e in range(config.Train.episodes):
        done = False
        state = env.reset()
        score = 0
        for t in range(agent.episode_step):
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)  # execute actions and wait until next scan(state)

            agent.store_transition(state, action, reward, next_state, done, shared_buffer, shared_lock)

            # if agent.memory.pointer >= agent.memory.capacity:
                # agent.learn()

            score += reward
            state = next_state

            if t >= 500:
                # rospy.loginfo("Time out!!")
                env.status = 'Time out'
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                print(f"||namespace: {namespace}||Ep: {e:3d}||score: {score:3.2f}||goal_dist: {env.goal_distance:.2f}||memory_capacity: {shared_buffer.pointer[0]:3d}/2k||steps: {t:3d}||status: {env.status}")
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
        # update local agent from global agent for each episode
        agent.load_state_dict(gAgent.state_dict())

def learn(gAgent, shared_buffer:Buffer):
    while(True):
        if shared_buffer.pointer[0] >= shared_buffer.capacity:
            gAgent.learn(shared_buffer)

if __name__ == '__main__':
    # import ipdb;ipdb.set_trace()
    # shared_buffer across processes
    shared_buffer = Buffer(Config.Memory.capacity, Config.Train.state_dim, Config.Train.device)
    shared_buffer.share_memory()

    shared_lock = Lock()
    num_processes = Config.num_processes

    # share the global parameters in multiprocessing
    gAgent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config.Train.device)
    gAgent.share_memory()

    processes = []
    # actor: num_processes, learner: 1
    for rank in range(num_processes+1):
        if rank != num_processes:
            p = mp.Process(target=train, args=(Config.namespace+str(rank), Config, gAgent, shared_buffer, shared_lock,))
        else:
            p = mp.Process(target=learn, args=(gAgent, shared_buffer,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
