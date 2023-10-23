import rospy
import argparse
import numpy as np
import pdb
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
from agent.policy import Agent
from env.env import Env
from util.costmap import state2costmap
from torch.utils.tensorboard import SummaryWriter
import random
import string

def train(args):
    # pdb.set_trace()
    rospy.init_node(args.namespace)

    if torch.cuda.is_available():
        device = args.device
    else:
        device = torch.device('cpu')

    # state_size = args.state_size
    action_size = args.action_size
    EPISODES = args.episodes

    env = Env(args)
    agent = Agent(action_size, episode_step=args.episode_step, replay_buffer_size=args.replay_buffer_size, device=device)
    scores, episodes = [], []

    #summary writer session name
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    writer = SummaryWriter(f'./log/{res}')
    # global_step = 0
    # start_time = time.time()

    for e in range(EPISODES):
        done = False
        state = env.reset()
        print(f'env is reset')
        state = state2costmap(state)
        # pdb.set_trace()

        score = 0
        for t in range(agent.episode_step):
            action = agent.select_action(state)

            # execute actions and wait until next scan(state)
            next_state, reward, done = env.step(action)
            next_state = state2costmap(next_state)

            agent.replay_buffer.push(state, action, reward, next_state, done)

            # if agent.replay_buffer.position >= agent.replay_buffer.capacity:
            if agent.replay_buffer.position >= 10 * args.batch_size:
                agent.update_parameters(agent.replay_buffer,args.batch_size)

            score += reward
            state = next_state

            if t == agent.episode_step-1:
                # rospy.loginfo("Time out!!")
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                print(f'Ep: {e}, '
                    f'score: {score:.5f}, '
                    f'memory_pointer: {agent.replay_buffer.position}/{agent.replay_buffer.capacity}, '
                    f'steps: {t}, '
                    f'epsilon: {agent.epsilon:.5f} '
                    f'status: {env.status}')
                break

        writer.add_scalar('reward', score, e)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=256)
    parser.add_argument('--action_size', type=int, default=2)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_buffer_size', type=int, default=5000)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    train(args)
