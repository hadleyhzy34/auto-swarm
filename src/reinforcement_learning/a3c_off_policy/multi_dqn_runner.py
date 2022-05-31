import rospy
import os
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
# from torch.utils.tensorboard import SummaryWriter 
from tensorboardX import SummaryWriter

writer = SummaryWriter()

def train(namespace, config, gAgent, shared_buffer,reward_shared_lock,g_lock):
    rospy.init_node(namespace)
    # initialize environment for each agent
    env = Env(namespace, config)

    agent = Agent(config.Train.state_dim, config.Train.action_dim, config.Train.device)
    # scores, episodes = [], []

    for e in range(config.Train.episodes):
        done = False
        state = env.reset()
        score = 0
        buffer_a = []
        buffer_s = []
        buffer_r = []
        for t in range(agent.episode_step):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)  # execute actions and wait until next scan(state)
            
            buffer_r.append(reward)
            buffer_s.append(state)
            buffer_a.append(action)

            score += reward
            state = next_state

            if t >= 500:
                # rospy.loginfo("Time out!!")
                env.status = 'Time out'
                done = True
            
            if (t+1) % config.Train.update_global_iter == 0 or done:
                # print(f'namespace: {namespace}, length of reward: {len(buffer_r[:pointer])}')
                agent.learn(gAgent, done, next_state, buffer_s, buffer_a, buffer_r, g_lock)
                buffer_r,buffer_a,buffer_s = [],[],[]
            
            if done:
                # scores.append(score)
                # episodes.append(e)
                print(f"||namespace: {namespace}||Ep: {e:3d}||score: {score:3.2f}||goal_dist: {env.goal_distance:.2f}||steps: {t:3d}||status: {env.status}")
                break
        
        shared_buffer.push_reward(score, reward_shared_lock)
        writer.add_scalar("episodic reward", shared_buffer.shared_reward.mean(), global_step=torch.div(shared_buffer.pointer, config.num_processes, rounding_mode = 'floor'))
        # writer.add_scalar(namespace + "Training_loss", loss, global_step=global_step)

if __name__ == '__main__':
    # shared_lock = Lock()
    reward_shared_lock = Lock()
    reward_buffer = Buffer()
    reward_buffer.share_memory()

    num_processes = Config.num_processes
    
    # share the global parameters in multiprocessing
    g_lock = Lock()
    gAgent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config.Train.device)
    gAgent.share_memory()
    # shared_opt = SharedAdam(gAgent.local_net.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    #
    processes = []
   
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(Config.namespace+str(rank), Config, gAgent,reward_buffer,reward_shared_lock,g_lock,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
