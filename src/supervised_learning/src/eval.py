import torch
import numpy as np
import torch.nn as nn
from agent.agent import Agent
from config.config import Config
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import rospy
import time
from env import Env
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def eval():
    env = Env('/tb3_0')
    agent = Agent(Config.Train.state_dim, Config.Train.action_dim, Config, mode = 'test', device=Config.Eval.device)

    # load pretrained model
    agent.load_state_dict(torch.load(Config.weight_file))
    agent.eval()

    # init ros node
    rospy.init_node('tb3_test', anonymous=True)

    while not rospy.is_shutdown():
        data = env.step()  #retrieve scanned data online
        # pdb.set_trace()
        # data = torch.cat([scan_data[None,:],env.goal_pose],dim=-1)  #(1,362)
        traj = agent.path_planning(data[None,:])  #traj planner, (1,l,2)
        # env.traj_data.data = traj[0].numpy()
        # pdb.set_trace()
        # convert (b,l,2) traj torch data to 1d list
        # pdb.set_trace()
        traj_data = traj[0].view(-1).numpy().tolist()  #(l*2)
        # data = Float32MultiArray(data = traj_data)
        # pdb.set_trace()

        # # visualize traj using marker data type
        # pdb.set_trace()
        m = Marker()
        
        m.header.frame_id = 'base_scan'
        m.id = 0
        m.type = Marker.POINTS
        m.color.a = 1.0
        m.action  = Marker.ADD

        m.scale.x = 0.1
        m.scale.y = 0.1
        m.color.g = 1.0

        for i in range(agent.length):
            p = Point(traj_data[i*2], traj_data[i*2+1], 0.)
            m.points.append(p)
        env.pub_traj.publish(m)

        # visualize using laserscan data type
        # data = LaserScan()
        # data.header.frame_id = 'base_scan'
        # data.ranges = traj_data
        # env.pub_traj.publish(data)
        # # env.pub_traj.publish(env.traj_data)
        


    # total_loss = []
    # dist_loss = []
    # coll_loss = []
    # angl_loss = []
    # # step_loss = []

    # for data in (pbar := tqdm(train_dataloader)):
    #     # pdb.set_trace()
    #     # data = data.to(Config.Train.device).float()
    #     traj = agent.path_planning(data)
    #     # action = agent.traj_follower(traj)
    #     # print(action)

    #     # next_state, reward, done = env.step(action)  # execute actions and wait until next scan(state)
                
    #     loss, loss_dist, loss_col, loss_angle = agent.learn(traj, data, train=False)
    #     total_loss.append(loss)
    #     dist_loss.append(loss_dist)
    #     coll_loss.append(loss_col)
    #     angl_loss.append(loss_angle)
    #     # step_loss.append(loss_step)

    #     pbar.set_description("loss: %.4f, dist: %.4f, coll: %.4f, angle: %.4f"%(statistics.fmean(total_loss),statistics.fmean(dist_loss),statistics.fmean(coll_loss),statistics.fmean(angl_loss)))
    #     pbar.refresh()
    #     # agent.scheduler.step()

if __name__ == '__main__':
    eval()
        
        