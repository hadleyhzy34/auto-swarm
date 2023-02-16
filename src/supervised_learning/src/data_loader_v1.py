import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pdb

class TensorDataset(Dataset):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # pdb.set_trace()
        self.file_list = np.genfromtxt(self.cfg.file_list, delimiter=',',dtype='str')  #(l,)
        self.length_list = np.genfromtxt(self.cfg.length_list, delimiter=',',dtype=np.int)  #(l,)

        #         cur_data = torch.from_numpy(data).view(-1,370).to(self.device)
        # for filename in os.listdir(data_dir):
        #     # pdb.set_trace()
        #     f = os.path.join(data_dir, filename)
        #     if os.path.isfile(f):
        #         self.file.append(f)
        #         self.file_count.append(self.total_length)
        #         # self.total_length += torch.load(f).view(-1,370).shape[0]
        #         if torch.load(f).shape[-1] != 370:
        #             pdb.set_trace()
        #         self.total_length += torch.load(f).shape[0]
        # # pdb.set_trace()

    def __getitem__(self, index):
        # pdb.set_trace()
        # check which file it belongs to
        for i,l in enumerate(self.length_list):
            if index < l:
                # pdb.set_trace()
                data = torch.load(self.file_list[i-1], map_location = self.device)
                # data = torch.load(self.file_list[i-1])
                # data = np.genfromtxt(self.file[i], delimiter=',')
                # self.data = torch.from_numpy(data).view(-1,370).to(self.device)
                # pdb.set_trace()
                index -= self.length_list[i-1]
                break
            if i == len(self.length_list) - 1:
                index = index - self.length_list[i]
                data = torch.load(self.file_list[i], map_location = self.device)
                # data = torch.load(self.file_list[i])
        # pdb.set_trace()
        assert index < data.shape[0], 'file shape is not correct'
        return data[index].float()  #(360+2+30)
        # goal_pose = data[index][360:361]  #(2,)
        
        # # exp_traj
        # exp_traj = data[index][-30:]

        # return torch.cat([data[index][:360], goal_pose, exp_traj],dim=-1).float()

    def __len__(self):
        return self.length_list[-1]
