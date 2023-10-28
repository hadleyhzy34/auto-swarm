import pdb
import argparse
import rospy
import numpy as np
import random
from std_msgs.msg import Float32MultiArray
import torch
# from agent.agent_dqn import Agent
from agent.agent import Agent
from env.env import Env
from torch.utils.tensorboard import SummaryWriter
import string

def train(args):
    rospy.init_node('turtlebot3_dqn_stage_1')

    if torch.cuda.is_available():
        device = args.device
    else:
        device = torch.device('cpu')

    #summary writer session name
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    writer = SummaryWriter(f'./log/{res}')

    env = Env(args.action_size)
    agent = Agent(args.state_size,
                  args.action_size,
                  episode_step=args.episode_step,
                  batch_size=args.batch_size,
                  memory_capacity=args.replay_buffer_size,
                  device=device)
    scores, episodes = [], []

    for e in range(args.episodes):
        done = False
        state = env.reset()
        score = 0.
        for t in range(agent.episode_step):
            action = agent.choose_action(state)

            # execute actions and wait until next scan
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)

            if agent.memory.pointer >= args.learn_threshold:
                agent.learn()

            score += reward
            state = next_state

            if t >= agent.episode_step:
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                print(f'Ep: {e}, '
                    f'score: {score:.5f}, '
                    f'memory_pointer: {agent.memory.pointer}/{agent.memory.capacity}, '
                    f'steps: {t}, '
                    f'epsilon: {agent.epsilon:.5f}, '
                    f'status: {env.status}'
                    )
                break

        writer.add_scalar('reward', score, e)

        if (e+1) % args.update_rank_frequency == 0:
            env.rank += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=362)
    parser.add_argument('--action_size', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--replay_buffer_size', type=int, default=5000)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--update_rank_frequency', type=int, default=100)
    parser.add_argument('--learn_threshold', type=int, default=1000)
    args = parser.parse_args()
    train(args)
