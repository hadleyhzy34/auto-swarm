#!/usr/bin/env python
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
import pdb
import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
import math
from math import pi
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import torch

class Env():
    def __init__(self, namespace='tb3_0', rank=0):
        self.goal_x = 0
        self.goal_y = 0
        
        self.action_size = 2
        self.namespace = namespace
        self.rank = rank
        self.theta = torch.zeros(2) # [yaw, omega]
        # self.initGoal = True
        self.get_goalbox = False  # reach goal or not
        self.pub_cmd_vel = rospy.Publisher(self.namespace+'/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber(self.namespace+'/odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # self.respawn_goal = Respawn()
        
        # reward weight
        self.r_wg = 2.5
        self.r_arrival = 500
        self.r_collision = -500
        self.r_rotation = -0.1
        # self.initGoal = True
        self.status = 'Running!!!'
        self.position = Pose()

        self.dist_range = 0.5

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
    
    def getState(self, scan):
        # import ipdb;ipdb.set_trace()
        scan_range = torch.empty((1,360))
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range[0,i] = 10.
            elif np.isnan(scan.ranges[i]):
                scan_range[0,i] = 0.
            else:
                scan_range[0,i] = scan.ranges[i]
        # pdb.set_trace()
        if min_range - scan_range.min() > 0:
            done = True # done because of collision

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # print(f'dist: {current_distance}')
        # if current_distance < 0.1:
        if current_distance < 0.2:
            # self.get_goalbox = True
            self.goal_reached = True
            done = True # done because of goal reached
        
        rel_goal_pose = torch.tensor([self.goal_x - self.position.x, self.goal_y - self.position.y])  #(2,)
        # pdb.set_trace()
        yaw = self.theta[None,0][None,:]  #(1,1)
        rot = torch.zeros((1,2,2))
        # rot[:,0,0] = np.cos(yaw)
        # rot[:,0,1] = np.sin(yaw)
        # rot[:,1,0] = - np.sin(yaw)
        # rot[:,1,1] = np.cos(yaw)
        # rel_goal_pose = torch.matmul(rot, rel_goal_pose[None,:,None]).squeeze(-1)  #(1,2)
        
        # print(f'goal: {torch.linalg.norm(rel_goal_pose)}, cur_dist: {current_distance}')
        return torch.cat([scan_range, yaw, rel_goal_pose[None,:]], dim = -1), done

    def step(self, action):
        # pdb.set_trace()
        # action = np.asarray(action)
        ang_vel = action[1].item()
        # max_angular_vel = 1.5
        # ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0].item()
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        # exec_time = time.time()

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass
        
        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        # pdb.set_trace()
        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return state, reward, done

    def setReward(self, state, done, action):
        """
        Description: set reward for each step
        args:
            state: (batch_size, state_dim)
            done: bool
            action: (batch_size, action_size)
        return:
            reward: value
        """
        reward = 0
        if done:
            # if self.get_goalbox:
            if self.goal_reached:
                reward += 500
                self.goal_reached = True
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Goal'
            else:
                reward -= 500
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Hit'
        else:
            goal_distance = np.linalg.norm([self.goal_x - self.position.x, self.goal_y - self.position.y])
            # reward += self.r_wg * (goal_distance - self.goal_distance) #wrong!!! previous_distance - current_distance
            reward += self.r_wg * (self.goal_distance - goal_distance).item()
            self.goal_distance = goal_distance
            if self.theta[1] > 0.7 or self.theta[1] < -0.7:
                reward += self.r_rotation * self.theta[1]

        return reward
       
    def reset(self):
        self.reset_model() #reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        # if self.initGoal:
        # self.goal_x, self.goal_y = self.respawn_goal.getPosition()
        # self.goal_x, self.goal_y = self.random_pts_map()
        # self.initGoal = False

        self.status = "Running!!!"
        self.goal_reached = False
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return state
    
    def reset_model(self):
        state_msg = ModelState()
        state_msg.model_name = self.namespace
        # update initial position and goal positions
        state_msg.pose.position.x, state_msg.pose.position.y, self.goal_x, self.goal_y = self.random_pts_map()
        # state_msg.pose.position.z = 0.3
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except (rospy.ServiceException) as e:
            print("gazebo/set_model_state Service call failed")
    
    def random_pts_map(self):
        """
        Description: random initialize starting position and goal position, make distance > 0.1
        return:
            x1,y1: initial position
            x2,y2: goal position
        """
        x1,x2,y1,y2 = 0,0,0,0
        while (x1 - x2) ** 2 < 0.01:
            x1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize x inside single map
            x2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        while (y1 - y2) ** 2 < 0.01:
            y1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize y inside single map
            y2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        x1 = -10 + (self.rank % 4) * 5 + x1
        y1 = -5 * (self.rank // 4) + y1

        x2 = -10 + (self.rank % 4) * 5 + x2
        y2 = -5 * (self.rank // 4) + y2
        
        dist = np.linalg.norm([y2 - y1, x2 - x1])
        while dist < self.dist_range - 0.3 or dist > self.dist_range:
        # while np.linalg.norm([y2 - y1, x2 - x1]) < 0.5:
            x2 = -10 + (self.rank % 4) * 5 + np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position
            y2 = -5 * (self.rank // 4) + np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position
            dist = np.linalg.norm([y2 - y1, x2 - x1])
            # pdb.set_trace()
            # print(f'current distnace is: {dist}') 
        return x1,y1,x2,y2
