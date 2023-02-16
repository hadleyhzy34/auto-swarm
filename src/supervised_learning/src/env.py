#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
import torch
import math
from math import pi
import time
import pdb
from visualization_msgs.msg import Marker
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Env():
    def __init__(self, namespace):
        self.namespace = namespace # namespace should be like 'tb3_0'

        self.goal_pose = torch.tensor([[4.8,0.]])  #(1,2)

        self.theta = np.zeros(2) # [yaw, omega]

        self.goal_reached = False

        # initial pose
        self.init_x = 0.
        self.init_y = 0.

        # initiate traj float array
        self.traj_data = Float32MultiArray()

        # reward weight
        self.r_wg = 2.5
        self.r_arrival = 500
        self.r_collision = -500
        self.r_rotation = -0.1
        # self.initGoal = True
        self.status = 'Running!!!'
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher(self.namespace+'/cmd_vel', Twist, queue_size=5)
        self.pub_traj = rospy.Publisher(self.namespace + '/traj', Marker, queue_size = 10)
        self.sub_odom = rospy.Subscriber(self.namespace+'/odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.theta[0] = yaw
        self.theta[1] = odom.twist.twist.angular.z
        # print(f'yaw,omega:{self.theta}')
    
    def getState(self, scan):
        # import ipdb;ipdb.set_trace()
        scan_range = np.array(scan.ranges)
        scan_range[np.isnan(scan_range)] = 10
        scan_range[np.isinf(scan_range)] = 10

        # heading = self.heading
        min_range = 0.2
        done = False
        
        if min_range > scan_range.min() > 0:
            done = True # done because of collision

        goal_distance = np.array([self.goal_x - self.position.x, self.goal_y - self.position.y])
        if np.linalg.norm(goal_distance) < 0.1:
            self.goal_reached = True
            done = True # done because of goal reached

        self.goal_dir = goal_distance / np.linalg.norm(goal_distance)
        return np.concatenate((scan_range, self.theta, self.goal_dir), axis=0), done
    
    def step(self):
        """data acquisitation for scan lidar data

        Returns:
            data: (362,), torch.float, 360 range data, 2 relative goal pose
        """
        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        # print(f'rostopic name is: {self.namespace + "/scan"}')
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass
        
        # remove abnormal data
        # test_data = np.float32(data.ranges)
        # pdb.set_trace()
        scan_range = np.float32(data.ranges)
        scan_range[np.isnan(scan_range)] = 10.
        scan_range[np.isinf(scan_range)] = 10.
        # scan_range = np.float32(scan_range)

        # pdb.set_trace()
        data = torch.from_numpy(scan_range)

        # print(f'data is subscribed {data.shape}')
        # print(f'curent position is: {self.position.x,self.position.y}')

        # robot frame based goal position
        rel_pose = self.goal_pose - torch.tensor([[self.position.x,self.position.y]])
        # 2.rotate based on robot frame
        # yaw = torch.tensor(self.theta[0]).float()
        rot = torch.tensor([[np.cos(self.theta[0]),np.sin(self.theta[0])],[-np.sin(self.theta[0]),np.cos(self.theta[0])]]).float()
        rel_pose = torch.mm(rot, rel_pose.transpose(0,1)).squeeze(-1)  #(2,)

        return torch.cat([data,rel_pose],dim=-1)