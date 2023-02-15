import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pdb
from config.config import Config
from expert_policy.a_start_test import AStarPlanner
import matplotlib.pyplot as plt

def preprocessing(data_dir, tgt_dir, file_list, length_list):
    """generate a file to record each file number of frames and total length

    Args:
        data_dir (string): file folder path
        tgt_dir (string): target directory to save pt files
    """
    total_length = 0

    for filename in os.listdir(data_dir):
        f = os.path.join(data_dir, filename)
        if os.path.isfile(f):
            data = np.genfromtxt(f, delimiter=',')
            print(f'loading file: {f}')
            # pdb.set_trace()
            cur_data = torch.from_numpy(data).view(-1,370)  #(b,370)
            # mask out initial point that is too closed to obstacles
            data_mask = torch.lt(cur_data[:,0:360],0.15).sum(-1)  #(b,)
            data_mask = torch.logical_not(data_mask)

            # check if goal position is too closed or too far from current robot position
            pose_dist = torch.linalg.norm(cur_data[:,-4:-2] - cur_data[:,-2:], dim=-1)  #(b,)
            dist_mask = torch.lt(pose_dist, 4.) * torch.gt(pose_dist, 0.15)  #(b,)
            cur_data = cur_data[data_mask*dist_mask]
            
            # generate expert policy
            grid_size = .1
            robot_radius = 0.07
            show_animation = True
            # ox, oy = [], []
            for batch_idx in range(cur_data.shape[0]):
                ox, oy = [], []
                sx = cur_data[batch_idx,-4].item()
                sy = cur_data[batch_idx,-3].item()
                gx = cur_data[batch_idx,-2].item()
                gy = cur_data[batch_idx,-1].item()
                for i in range(360):
                    if cur_data[batch_idx,i] < 3.5:
                        ox.append((np.cos(i * np.pi / 180) * cur_data[batch_idx,i]).item())
                        oy.append((np.sin(i * np.pi / 180) * cur_data[batch_idx,i]).item())
                if show_animation:  # pragma: no cover
                    plt.plot(ox, oy, ".k")
                    plt.plot(sx, sy, "og")
                    plt.plot(gx, gy, "xb")
                    plt.grid(True)
                    plt.axis("equal")
                
                pdb.set_trace()
                a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
                rx, ry = a_star.planning(sx, sy, gx, gy)
                # pdb.set_trace()

                if show_animation:  # pragma: no cover
                    plt.plot(rx, ry, "-r")
                    plt.pause(0.001)
                    plt.show()

            # pdb.set_trace()
            tgt_file = tgt_dir + '/' + filename[:-4] + '.pt'
            torch.save(cur_data.view(-1,370), tgt_file)

            with open(file_list,'a') as f:
                np.savetxt(f, np.array([tgt_file]), fmt='%s',delimiter=',')
            
            with open(length_list,'a') as f:
                np.savetxt(f, np.array([total_length]), fmt='%d',delimiter=',')
            
            total_length += cur_data.shape[0]

if __name__ == '__main__':
    preprocessing(Config.data_file, Config.tgt_file, Config.file_list, Config.length_list)
