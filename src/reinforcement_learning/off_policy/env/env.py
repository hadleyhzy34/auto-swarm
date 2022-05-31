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
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False  # reach goal or not
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # self.respawn_goal = Respawn()

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        # print(f'odometry is called')
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        # import ipdb;ipdb.set_trace()
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True # done because of collision

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True
            done = True # done because of goal reached

        # print(f'current agent position: {self.position.x}, {self.position.y}')
        return scan_range + [heading, current_distance], done

    def setReward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:
            if self.get_goalbox: # done because of goal reached
                rospy.loginfo("Goal!")
                reward = 200
                self.pub_cmd_vel.publish(Twist())
                self.get_goalbox = False
            else:  # done because of collision
                rospy.loginfo("Collision!!")
                reward = -200
                self.pub_cmd_vel.publish(Twist())

        # if done:
        #     rospy.loginfo("Collision!!")
        #     reward = -200
        #     self.pub_cmd_vel.publish(Twist())

        # if self.get_goalbox:
        #     rospy.loginfo("Goal!!")
        #     reward = 200
        #     self.pub_cmd_vel.publish(Twist())
        #     # self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
        #     # self.goal_x, self.goal_y = 1.,1.
        #     self.goal_distance = self.getGoalDistace()
        #     self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        # exec_time = time.time()

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        
        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        self.reset_model() #reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            # self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.goal_x, self.goal_y = self.r_pts_map1()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)
    
    def reset_model(self):
        state_msg = ModelState()
        state_msg.model_name = 'autorace'
        state_msg.pose.position.x, state_msg.pose.position.y = self.r_pts_map1()
        # state_msg.pose.position.z = 0.3
        # state_msg.pose.orientation.x = 0
        # state_msg.pose.orientation.y = 0
        # state_msg.pose.orientation.z = 0
        # state_msg.pose.orientation.w = 1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except (rospy.ServiceException) as e:
            print("gazebo/set_model_state Service call failed")
    
    def r_pts_map1(self):
        x = np.random.uniform(0.14, 1-0.14)
        y = np.random.uniform(0.14, 1-0.14)
        while True:
            idx = np.random.randint(36)
            if idx not in [7,8,10,16,19,25,26,28]:
                break
        x = -3 + (idx%6) + x
        y = 3 - (idx//6) -1 +y
        return x,y
