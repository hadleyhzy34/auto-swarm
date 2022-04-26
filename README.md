# Quick Start within 3 Minutes 
Compiling tests passed on ubuntu **16.04, 18.04, and 20.04** with ros installed.
You can just execute the following commands one by one.
```
sudo apt-get install libarmadillo-dev
git clone -b stable https://github.com/hadleyhzy34/auto-swarm.git
cd auto-swarm
catkin_make -j$(nproc)
source devel/setup.bash
```

# Environment

ROS --version 18.04

# Installation
## ros(required)
```
sudo apt-get install ros-<distro>-desktop-full ros-<distro>-roscpp ros-<distro>-rospy ros-<distro>-nav-msgs ros-<distro>-nav-msgs
sudo apt install ros-<distro>-costmap-2d \
ros-<distro>-move-base \
ros-<distro>-global-planner \
ros-<distro>-amcl
ros-<distro>-ackermann-msgs
```

## ros dependencies(required)
go to top of the folder and execute this command

```
rosdep install --from-paths src --ignore-src -r -y
```

# Components && Applications

## end to end

### Reinforcement Learning

* on-policy
* off-policy

### Imitation Learning

* behavior cloning
* inverse reinforcement learning

## Localization && Mapping

### Localization
* lidar odometry
* vision odometry
* mapping localization

### Lidar Mapping
* lego-loam

### Vision Mapping

## Perception

## Sensor Fusion

## Global Planner

## Local Planner

* RL path planner
* ego planner
* orca
* mpc

## Controller

### Pure Pursuit
### PID
### LQR + PID
### MPC(IPOPT/C++)


# Simulation

## quadcopter rviz simulation

initialize rviz simulator
```
source devel/setup.bash
roslaunch ego_planner rviz.launch
```

initialize single drone without making any decisions
```
source devel/setup.bash
roslaunch ego_planner swarm_test.launch
```

## Gazebo Simulation
### turtlebot3 simulation
#### folder

```
src/simulation/turtlebot3*
```

#### instructions

```
vim ~/.bashrc
export TURTLEBOT3_MODEL=burger
source ~/.bashrc
```

* launch world env
```
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

* launch rviz visualization
```
roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch
```

* launch teleop key
```
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

* lanuch multi-robot world env
```
roslaunch turtlebot3_gazebo multi_turtlebot3.launch
```

![multi-robot](https://github.com/hadleyhzy34/auto-swarm/blob/main/src/simulation/turtlebot3/screen_shot.png)

# Deployment

* GPU/CUDA
* TensoRT

# How to make contributions

## create a new application package

```
catkin_create_pkg mpc roscpp rospy std_msgs
```

# Acknowledgements
