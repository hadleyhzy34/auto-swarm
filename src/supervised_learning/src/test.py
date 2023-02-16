import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from agent.agent import Agent
from config.config import Config
# from data_loader_v0 import TensorDataset
# from data_loader import TensorDataset
from data_loader_v1 import TensorDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics

def test():
    data = TensorDataset(Config, device=Config.Train.device)
    # data = TensorDataset(Config.data_file, device=Config.Test.device)
    agent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config, mode='test', device=Config.Test.device)

    # load pretrained model
    agent.load_state_dict(torch.load(Config.weight_file))
    agent.eval()

    train_dataloader = DataLoader(data, batch_size=Config.Test.batch_size, shuffle=True)

    total_loss = []
    dist_loss = []
    coll_loss = []
    angl_loss = []
    # step_loss = []

    print(f"current learning rate: {agent.optimizer.param_groups[0]['lr']}")
    for data in (pbar := tqdm(train_dataloader)):
        # pdb.set_trace()
        # data = data.to(Config.Train.device).float()
        traj, traj_ineq = agent.path_planning(data)
        # action = agent.traj_follower(traj)
        # print(action)

        # next_state, reward, done = env.step(action)  # execute actions and wait until next scan(state)
        # pdb.set_trace()                
        loss, loss_dist, loss_col, loss_angle = agent.learn(traj, traj_ineq, data, train=False)
        total_loss.append(loss)
        dist_loss.append(loss_dist)
        coll_loss.append(loss_col)
        angl_loss.append(loss_angle)
        # step_loss.append(loss_step)

        pbar.set_description("loss: %.4f, dist: %.4f, coll: %.4f, angle: %.4f"%(statistics.fmean(total_loss),statistics.fmean(dist_loss),statistics.fmean(coll_loss),statistics.fmean(angl_loss)))
        pbar.refresh()
        # agent.scheduler.step()

if __name__ == '__main__':
    test()
