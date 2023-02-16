import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pdb
from config.config import Config
from expert_policy.a_start_test import AStarPlanner
from expert_policy.bspline import Spline2D
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
            print(f'trying to load file: {f}')
            data = np.genfromtxt(f, delimiter=',')
            print(f'loading file: {f}')
            # pdb.set_trace()
            cur_data = torch.from_numpy(data).view(-1,370).to(torch.float32)  #(b,370)
            # pdb.set_trace()
            # mask out initial point that is too closed to obstacles
            data_dist = torch.min(cur_data[:,0:360],dim=-1).values  #(b,)
            data_mask = torch.gt(data_dist, 0.15)  #(b,)
            
            # goal position based on robot fame
            yaw = cur_data[:,360]  #(b,)
            rot = torch.zeros((cur_data.shape[0],2,2))
            rot[:,0,0] = torch.cos(yaw)
            rot[:,0,1] = torch.sin(yaw)
            rot[:,1,0] = - torch.sin(yaw)
            rot[:,1,1] = torch.cos(yaw)
            rel_goal_pose = cur_data[:,-2:] - cur_data[:,-4:-2]
            # pdb.set_trace()
            goal_pose = torch.matmul(rot, rel_goal_pose[:,:,None]).squeeze(-1)  #(b,2)
            # goal_pose = torch.matmul(rel_goal_pose[:,None,:], rot).squeeze(1)  #(b,2)
            
            # obstacles in 2d axis frame
            # pdb.set_trace()
            rad_points = torch.zeros((cur_data.shape[0], 360, 2))
            # pdb.set_trace()
            rad_points[:,:,0] = torch.cos((torch.arange(0,360)) * torch.pi / 180) * cur_data[:,0:360]
            rad_points[:,:,1] = torch.sin((torch.arange(0,360)) * torch.pi / 180) * cur_data[:,0:360]
        
            # check if goal position is too closed to obstacles
            goal_dist = torch.linalg.norm(rad_points - goal_pose[:,None,:], dim=-1).min(-1).values  #(b,360)
            goal_mask = torch.gt(goal_dist, 0.15)  #(b,)
            # goal_mask = torch.logical_not(torch.lt(goal_mask, 0.15).sum(-1) > 0)  #(b,)
            
            # check if goal position is too closed or too far from current robot position
            pose_dist = torch.linalg.norm(cur_data[:,-4:-2] - cur_data[:,-2:], dim=-1)  #(b,)
            dist_mask = torch.lt(pose_dist, 4.) * torch.gt(pose_dist, 0.5)  #(b,)
            cur_data = cur_data[data_mask * dist_mask * goal_mask]
            goal_pose = goal_pose[data_mask * dist_mask * goal_mask]
            
            # pdb.set_trace()
            # res_data = torch.zeros((cur_data.shape[0],cur_data.shape[1]+30))
            # res_data = torch.zeros((1,360+2+30))
            # res_data[:,:-30] = cur_data
            res_data = None

            # generate expert policy
            grid_size = .1
            robot_radius = 0.075
            show_animation = False
            # ox, oy = [], []
            for batch_idx in range(0, cur_data.shape[0], 10):
                temp_data = torch.zeros(360+2+30)
                temp_data[:360] = cur_data[batch_idx][:360]
                ox, oy = [], []
                
                sx = 0.
                sy = 0.
                
                gx = goal_pose[batch_idx,0].item()
                gy = goal_pose[batch_idx,1].item()
                temp_data[360] = gx
                temp_data[361] = gy

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
                
                if batch_idx == 1:
                    pdb.set_trace()
                
                a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
                rx, ry = a_star.planning(sx, sy, gx, gy)  #([n,],[n,]
                if len(rx) <= 1:
                    plt.plot(ox, oy, ".k")
                    plt.plot(sx, sy, "og")
                    plt.plot(gx, gy, "xb")
                    plt.grid(True)
                    plt.axis("equal")
                    plt.show()
                sp = Spline2D(rx,ry)
                ts = np.arange(0, sp.s[-1],sp.s[-1]/15)
                ts = ts[:15]
                if len(ts) != 15:
                    pdb.set_trace()
                assert len(ts) == 15, 'length of ts is not equal to 15'

                # pdb.set_trace()
                test_x, test_y = [], []
                # factor = len(rx) // 15
                # pdb.set_trace()
                for iss in ts:
                    ix, iy = sp.calc_position(iss)
                    test_x.append(ix)
                    test_y.append(iy)
                
                # pdb.set_trace()
                for i in range(15):
                    temp_data[-i-15-1] = test_x[i]
                    temp_data[-i-1] = test_y[i]
                
                if show_animation:  # pragma: no cover
                    plt.plot(test_x, test_y, "-r")
                    plt.pause(0.001)
                    plt.show()
                
                if res_data == None:
                    res_data = temp_data[None,:]  #(1,360+2+30)
                else:
                    res_data = torch.cat([res_data, temp_data[None,:]],dim=0)  #(b,360+2+30)
                
                # pdb.set_trace()
                if show_animation:
                    dx,dy= [],[]
                    for i in range(360):
                        if res_data[-1,i] < 3.5:
                            dx.append((np.cos(i * np.pi / 180) * res_data[-1,i]).item())
                            dy.append((np.sin(i * np.pi / 180) * res_data[-1,i]).item())
                    plt.plot(dx, dy, ".k")
                    plt.grid(True)
                    plt.axis("equal")

                    for j in range(15):
                        plt.scatter(res_data[-1,-30+j].detach().cpu().numpy(), res_data[-1,-15+j].detach().cpu().numpy(), color='purple')

                    plt.scatter([0.], [0.], color='green')
                    
                    # pdb.set_trace()
                    plt.scatter(res_data[-1,360].cpu().numpy(), res_data[-1,361].cpu().numpy(),color='red')
                    plt.show()

            # pdb.set_trace()
            tgt_file = tgt_dir + '/' + filename[:-4] + '.pt'
            # pdb.set_trace()
            # torch.save(res_data.view(-1,400), tgt_file)
            torch.save(res_data, tgt_file)
            # torch.save(cur_data.view(-1,370), tgt_file)

            with open(file_list,'a') as f:
                np.savetxt(f, np.array([tgt_file]), fmt='%s',delimiter=',')
            
            with open(length_list,'a') as f:
                np.savetxt(f, np.array([total_length]), fmt='%d',delimiter=',')
            
            # total_length += cur_data.shape[0]
            total_length += res_data.shape[0]

if __name__ == '__main__':
    preprocessing(Config.data_file, Config.tgt_file, Config.file_list, Config.length_list)
