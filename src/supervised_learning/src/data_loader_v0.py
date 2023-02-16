import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pdb

class TensorDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data = None
        self.device = device
        self.file = []
        self.file_count = []
        self.total_length = 0
        for filename in os.listdir(data_dir):
            # pdb.set_trace()
            f = os.path.join(data_dir, filename)
            if os.path.isfile(f):
                self.file.append(f)
                self.file_count.append(self.total_length)
                # self.total_length += torch.load(f).view(-1,370).shape[0]
                if torch.load(f).shape[-1] != 370:
                    pdb.set_trace()
                self.total_length += torch.load(f).shape[0]
        # pdb.set_trace()

    def __getitem__(self, index):
        # check which file it belongs to
        for i,f in enumerate(self.file_count):
            if index < f:
                data = torch.load(self.file[i-1], map_location = self.device)
                # data = np.genfromtxt(self.file[i], delimiter=',')
                # self.data = torch.from_numpy(data).view(-1,370).to(self.device)
                # pdb.set_trace()
                index -= self.file_count[i-1]
                break
            if i == len(self.file_count) - 1:
                index = index - self.file_count[i]
                data = torch.load(self.file[i], map_location = self.device)
        # pdb.set_trace()
        # index = 
        # set position of target point with reference to current robot frame
        # 1. current position with reference to world frame
        # pdb.set_trace()
        # cur_pose_w = self.data[index][-4:-2] + self.data[index][-6:-4]
        goal_pose = data[index][-2:] - data[index][-4:-2]
        # 2.rotate based on robot frame
        yaw = data[index][360]
        rot = torch.tensor([[torch.cos(yaw),torch.sin(yaw)],[-torch.sin(yaw),torch.cos(yaw)]]).to(self.device)
        goal_pose = torch.mm(rot, goal_pose[:,None]).squeeze(-1)  #(2,)

        return torch.cat([data[index],goal_pose],dim=-1).float()

    def __len__(self):
        return self.total_length