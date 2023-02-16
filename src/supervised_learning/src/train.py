import imp
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
import time

def train():
    torch.autograd.set_detect_anomaly(True)
    # file = "/home/hadley/Development/self-supervised-recurrent-path-planning/data/csv"
    # file = "/Users/hadley/Developments/self-supervised-recurrent-path-planning/data/csv"
    # data = TensorDataset(Config.data_file, device=Config.Train.device)
    data = TensorDataset(Config, device=Config.Train.device)

    # agent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config, device=Config.Train.device)
    agent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config, device=Config.Train.device)

    # load pretrained model
    # agent.load_state_dict(torch.load(Config.weight_file))
    agent.train()
    
    train_dataloader = DataLoader(data, batch_size=Config.Train.batch_size, shuffle=True)

    total_all_loss = []
    total_dist_loss = []
    total_coll_loss = []
    total_angl_loss = []
    total_bspline_loss = []
    # total_step_loss = []

    for i in range(Config.Train.episodes):
        print(f"current learning rate: {agent.optimizer.param_groups[0]['lr']}")
        total_loss = []
        dist_loss = []
        coll_loss = []
        angl_loss = []
        bspline_loss = []
        # step_loss = []
        
        # # add obs loss after 5 epochs
        # if i == 5:
        #     agent.obs_loss = True

        for data in (pbar := tqdm(train_dataloader)):
            # pdb.set_trace()
            # data = data.to(Config.Train.device).float()
            traj = agent.path_planning(data)

            # loss, loss_dist, loss_col, loss_angle, loss_bspline = agent.learn(traj, data, train=True)
            loss, loss_dist, loss_col, loss_angle = agent.learn(traj, data, train=True)
            total_loss.append(loss)
            dist_loss.append(loss_dist)
            coll_loss.append(loss_col)
            angl_loss.append(loss_angle)
            # bspline_loss.append(loss_bspline)
            # step_loss.append(loss_step)
    
            pbar.set_description("loss: %.4f, dist: %.4f, coll: %.4f, angle: %.4f"%(statistics.fmean(total_loss),statistics.fmean(dist_loss),statistics.fmean(coll_loss),statistics.fmean(angl_loss)))
            # pbar.set_description("loss: %.4f, dist: %.4f, coll: %.4f, angle: %.4f, bspline: %.4f"%(statistics.fmean(total_loss),statistics.fmean(dist_loss),statistics.fmean(coll_loss),statistics.fmean(angl_loss),statistics.fmean(bspline_loss)))
            pbar.refresh()

        # t = time.localtime()
        # current_time = time.strftime("%H_%M_%S", t)
        # torch.save(agent.state_dict(), Config.weight_folder+current_time+'.pt')
        torch.save(agent.state_dict(), Config.weight_folder + str(i) + '.pth')

        # if i > 10:
            # agent.scheduler.step()
        agent.scheduler.step()

    # plt.plot(np.arange(len(total_all_loss)),total_all_loss,label="total_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_dist_loss,label="dist_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_coll_loss,label="coll_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_angl_loss,label="angle_loss")
    # # plt.plot(np.arange(len(total_all_loss)),total_step_loss,label="step_loss")
    # plt.legend()
    # plt.show()

    torch.save(agent.state_dict(), Config.weight_file)

if __name__ == '__main__':
    train()
        
        
