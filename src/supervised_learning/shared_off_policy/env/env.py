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
import math
from math import pi
import time
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Env():
    def __init__(self, namespace, config):
        self.namespace = namespace # namespace should be like 'tb3_0'
        self.rank = int(namespace[-1]) # id of robots
        self.goal_x = 0
        self.goal_y = 0
        # self.heading = 0
        # self.yaw = np.zeros(1) # robot yaw angle
        self.theta = np.zeros(2) # [yaw, omega]
        self.action_size = config.Train.action_dim
        self.goal_reached = False

        # reward weight
        self.r_wg = 2.5
        self.r_arrival = 500
        self.r_collision = -500
        self.r_rotation = -0.1
        # self.initGoal = True
        self.status = 'Running!!!'
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher(self.namespace+'/cmd_vel', Twist, queue_size=5)
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
        scan_range[np.isnan(scan_range)] = 3.5
        scan_range[np.isinf(scan_range)] = 3.5

        # heading = self.heading
        min_range = 0.13
        done = False
        
        if min_range > scan_range.min() > 0:
            done = True # done because of collision

        goal_distance = np.array([self.goal_x - self.position.x, self.goal_y - self.position.y])
        if np.linalg.norm(goal_distance) < 0.1:
            self.goal_reached = True
            done = True # done because of goal reached

        self.goal_dir = goal_distance / np.linalg.norm(goal_distance)
        return np.concatenate((scan_range, self.theta, self.goal_dir), axis=0), done

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
            if self.goal_reached:
                reward += 15
                self.goal_reached = True
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Goal'
            else:
                reward -= 15
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Hit'
        else:
            goal_distance = np.linalg.norm([self.goal_x - self.position.x, self.goal_y - self.position.y])
            # reward += self.r_wg * (goal_distance - self.goal_distance) #wrong!!! previous_distance - current_distance
            reward += self.r_wg * (self.goal_distance - goal_distance)
            self.goal_distance = goal_distance
            if self.theta[1] > 0.7 or self.theta[1] < -0.7:
                reward += self.r_rotation * self.theta[1]

        return reward
       
    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass
        
        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

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
        return np.asarray(state)
    
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

        while np.linalg.norm([y2 - y1, x2 - x1]) < 0.5:
            x2 = -10 + (self.rank % 4) * 5 + x2
            y2 = -5 * (self.rank // 4) + y2
        
        return x1,y1,x2,y2
