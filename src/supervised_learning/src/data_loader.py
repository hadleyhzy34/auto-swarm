import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pdb

class TensorDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data = None
        self.device = device
        for filename in os.listdir(data_dir):
            f = os.path.join(data_dir, filename)
            if os.path.isfile(f):
                print(f'loading file: {f}')
                data = np.genfromtxt(f, delimiter=',')
                cur_data = torch.from_numpy(data).view(-1,370).to(self.device)
                data_mask = torch.lt(cur_data[:,0:360],0.15).sum(-1)  #(b,)
                data_mask = torch.logical_not(data_mask)
                cur_data = cur_data[data_mask]

                if self.data == None:
                    self.data = torch.from_numpy(data).view(-1,370).to(self.device)
                else:
                    self.data = torch.concat([self.data, torch.from_numpy(data).view(-1,370).to(self.device)],dim=0)
                # break

    def __getitem__(self, index):
        # set position of target point with reference to current robot frame
        # 1. current position with reference to world frame
        # pdb.set_trace()
        # cur_pose_w = self.data[index][-4:-2] + self.data[index][-6:-4]
        goal_pose = self.data[index][-2:] - self.data[index][-4:-2]
        # 2.rotate based on robot frame
        yaw = self.data[index][360]
        rot = torch.tensor([[torch.cos(yaw),torch.sin(yaw)],[-torch.sin(yaw),torch.cos(yaw)]]).to(self.device)
        goal_pose = torch.mm(rot, goal_pose[:,None]).squeeze(-1)  #(2,)

        return torch.cat([self.data[index],goal_pose],dim=-1).float()

        # theta = torch.arctan((s[:,-1] - s[:,-3]) / (s[:,-2] - s[:,-4]))
        # theta_prime = theta - s[:,361]
        # abs_length = torch.linalg.norm(torch.cat([(s[:,-2] - s[:,-4])[:,None], (s[:,-1] - s[:,-3])[:,None]],dim=-1),dim=-1)  #(b,)
        # # target position given current robot frame
        # tgt_pose = abs_length[:,None] * torch.cat([torch.cos(theta_prime)[:,None],torch.sin(theta_prime)[:,None]],dim=-1)  #(b,2)

        # return self.data[index]

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    # 生成数据
    file = "/Users/hadley/Developments/self-supervised-recurrent-path-planning/data/csv"
    data = TensorDataset(file)
    
    pdb.set_trace()
    train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
    
    for data in train_dataloader:
        print(data)
    