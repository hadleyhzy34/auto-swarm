import rospy
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import numpy as np
import math
from math import pi
import time
import torch
import matplotlib.pyplot as plt
import pdb
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Env():
    def __init__(self, args):
        self.rank = 0
        self.goal_x = 0
        self.goal_y = 0
        self.map_x = -10.
        self.map_y = -10.
        self.goal_reached = False
        self.device = args.device
        self.namespace = args.namespace
        self.theta = np.zeros(2) # [yaw, omega]
        self.heading = 0
        self.action_size = args.action_size
        self.initGoal = True
        self.get_goalbox = False  # reach goal or not
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher(self.namespace+'/cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber(self.namespace+'/odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        # self.respawn_goal = Respawn()

        # reward weight
        self.theta_diff = None
        self.r_wg = 2.5
        self.r_arrival = 15
        self.r_collision = -15
        self.r_rotation = -0.1

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        # print(f'odometry is called')
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.theta[0] = yaw
        self.theta[1] = odom.twist.twist.angular.z

    def getState(self, scan):
        # import ipdb;ipdb.set_trace()
        scan_range = np.array(scan.ranges)
        scan_range[np.isnan(scan_range)] = 10.
        scan_range[np.isinf(scan_range)] = 10.

        min_range = 0.13
        done = False

        if min_range > scan_range.min():
            # print(f'collision: {scan_range.min()}')
            done = True # done because of collision

        goal_pos = np.array([self.goal_x - self.position.x, self.goal_y - self.position.y])
        self.goal_distance = np.linalg.norm(goal_pos)
        
        if self.goal_distance < 0.2:
            self.goal_reached = True
            done = True # done because of goal reached

        # relative goal position based on robot base frame
        yaw = self.theta[0]
        rot = np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])  #(2,2)
        goal_pos = np.matmul(rot, goal_pos[:,None])[:,0]  #(2,)
        
        # print(f'cur pos: {self.position.x,self.position.y}, goal_pos: {self.goal_x,self.goal_y}')
        return np.concatenate((scan_range, goal_pos), axis=0), done

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
                self.goal_reached = False
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Goal'
            else:
                reward -= 15
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Hit'
        else:
            goal_distance = np.linalg.norm([self.goal_x - self.position.x, self.goal_y - self.position.y])
            
            # # goal distance reward
            reward += self.r_wg * (self.goal_distance - goal_distance)
            self.goal_distance = goal_distance
            
            # punish and accumulate number of steps
            # reward -= .1
            
            # punish large angular acceleration
            if self.theta[1] > 0.7 or self.theta[1] < -0.7:
                reward += self.r_rotation * self.theta[1]

        return reward
    
    def setSacReward(self, state, done, action):
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
                self.goal_reached = False
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Goal'
            else:
                reward -= 15
                self.pub_cmd_vel.publish(Twist())
                self.status = 'Hit'
        else:
            goal_distance = np.linalg.norm([self.goal_x - self.position.x, self.goal_y - self.position.y])
            
            # # goal distance reward
            if self.goal_distance - goal_distance > 0:
                reward += self.goal_distance - goal_distance
            else:
                reward += 2 * (self.goal_distance - goal_distance)
            self.goal_distance = goal_distance
            
            # goal direction reward/penalty
            # pdb.set_trace()
            # self.theta[0] in range of [-pi,pi]
            theta_diff = np.abs(self.theta[0] - self.goal_theta)
            if theta_diff > self.theta_diff:
                reward += 2 * (self.theta_diff - theta_diff)
            else:
                reward += self.theta_diff - theta_diff
            self.theta_diff = theta_diff
            
            # punish and accumulate number of steps
            # reward -= .1
            
            # punish large angular acceleration
            # if self.theta[1] > 0.7 or self.theta[1] < -0.7:
            #     reward += self.r_rotation * self.theta[1]

        return reward
    
    def step(self, action):
        # pdb.set_trace()
        # max_angular_vel = 1.5
        # ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.pub_cmd_vel.publish(vel_cmd)
        # exec_time = time.time()

        # waiting more or less equal to 0.2 s until scan data received, stable behavior
        # pdb.set_trace()
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        # print(f'waiting scan data duration is: {time.time() - exec_time}')
        state, done = self.getState(data)
        # reward = self.setReward(state, done, action)
        reward = self.setSacReward(state, done, action)

        return state, reward, done
        # return np.asarray(state), reward, done

    def reset(self):
        # pdb.set_trace()
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        #
        self.reset_model() #reinitialize model starting position

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(self.namespace+'/scan', LaserScan, timeout=5)
            except:
                pass

        self.status = "Running!!!"
        self.goal_reached = False
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        # set goal theta and initial goal angle diff
        self.goal_theta = np.arctan2(self.goal_y-self.init_y,self.goal_x-self.init_x)
        self.theta_diff = np.abs(self.theta[0]-self.goal_theta)

        if self.theta_diff > np.pi * 2:
            raise Exception("theta diff is out of range")

        return state
        # return np.asarray(state)
 
    def reset_model(self):
        state_msg = ModelState()
        state_msg.model_name = self.namespace
        # update initial position and goal positions
        state_msg.pose.position.x, state_msg.pose.position.y, self.goal_x, self.goal_y = self.random_pts_map()
        self.init_x = state_msg.pose.position.x
        self.init_y = state_msg.pose.position.y
        # state_msg.pose.position.z = 0.3
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 1

        # modify target cricket ball position
        target = ModelState()
        target.model_name = 'cricket_ball'
        target.pose.position.x = self.goal_x
        target.pose.position.y = self.goal_y

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )
            res  = set_state( target )

        except (rospy.ServiceException) as e:
            print("gazebo/set_model_state Service call failed")
    
    def random_pts_map(self):
        """
        Description: random initialize starting position and goal position, make distance > 0.1
        return:
            x1,y1: initial position
            x2,y2: goal position
        """
        # pdb.set_trace()
        # print(f'goal position is going to be reset')
        x1,x2,y1,y2 = 0,0,0,0
        while (x1 - x2) ** 2 < 0.01:
            x1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize x inside single map
            x2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        while (y1 - y2) ** 2 < 0.01:
            y1 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize y inside single map
            y2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

        x1 = self.map_x + (self.rank % 4) * 5 + x1
        y1 = self.map_y + (3 - self.rank // 4) * 5 + y1

        x2 = self.map_x + (self.rank % 4) * 5 + x2
        y2 = self.map_y + (3 - self.rank // 4) * 5 + y2

        # set waypoint within scan range
        # pdb.set_trace()
        dist = np.linalg.norm([y2 - y1, x2 - x1]) 
        while dist < 0.25 or dist > 3.5:
            x2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position
            y2 = np.random.randint(5) + np.random.uniform(0.16, 1-0.16) # random initialize goal position

            x2 = self.map_x + (self.rank % 4) * 5 + x2
            y2 = self.map_y + (3 - self.rank // 4) * 5 + y2
            dist = np.linalg.norm([y2 - y1, x2 - x1])

        # self.rank = (self.rank + 1) % 8
        # print(f'goal position is respawned')

        return x1,y1,x2,y2
