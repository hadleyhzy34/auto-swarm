# Quick Start within 3 Minutes 
Compiling tests passed on ubuntu **16.04, 18.04, and 20.04** with ros installed.
You can just execute the following commands one by one.
```
sudo apt-get install libarmadillo-dev
sudo apt-get install ros-<distro>-gazebo-ros
git clone -b stable https://github.com/hadleyhzy34/auto-swarm.git
cd auto-swarm
catkin_make -j$(nproc)
source devel/setup.bash
```

# Environment

ROS --version 18.04, 20.04

# Components && Applications

## end to end

### Reinforcement Learning

* on-policy
* off-policy

### Imitation Learning

* behavior-cloning
* inverse reinforcement learning

# Simulation

## Gazebo Simulation
### turtlebot3 simulation
#### folder

```
src/simulation/turtlebot3*
```

#### installation

```
sudo apt-get install ros-<distro>-gazebo-ros-pkgs ros-<distro>-gazebo-ros-control
```

#### instructions

```
vim ~/.bashrc
export TURTLEBOT3_MODEL=burger
source ~/.bashrc
```

#### scenarios

* turtlebot3 + obstacles

```
roslaunch turtlebot3_gazebo turtlebot3_single_map0.launch
```

* multi-turtlebot3 + multi-maps

```
roslaunch turtlebot3_gazebo turtlebot3_multi_map0.launch
```

##### setup

1.map

```
<include file="$(find gazebo_ros)/launch/empty_world.launch">
  <arg name="world_name" value="$(find turtlebot3_gazebo)/worlds/turtlebot3_multi_map0.world"/>
  <arg name="paused" value="false"/>
  <arg name="use_sim_time" value="true"/>
  <arg name="gui" value="true"/>
  <arg name="headless" value="false"/>
  <arg name="debug" value="false"/>
</include>  
```


2.URDF model file

```
<group ns = "$(arg 0_tb3)">
    <param name="robot_description" command="$(find xacro)/xacro --inorder $(find turtlebot3_description)/urdf/turtlebot3_$(arg model)_for_autorace_2020.urdf.xacro" />

    <node pkg="robot_state_publisher" type="robot_state_publisher" name="robot_state_publisher" output="screen">
      <param name="publish_frequency" type="double" value="50.0" />
      <param name="tf_prefix" value="$(arg 0_tb3)" />
    </node>
    
    <node name="spawn_urdf" pkg="gazebo_ros" type="spawn_model" args="-urdf -model $(arg 0_tb3) -x $(arg 0_tb3_x_pos) -y $(arg 0_tb3_y_pos) -z $(arg 0_tb3_z_pos) -Y $(arg 0_tb3_yaw) -param robot_description" />
  </group>
```

3.namespace and initial pose:

```
<arg name="0_tb3"  default="tb3_0"/>
<arg name="1_tb3"  default="tb3_1"/>
<arg name="2_tb3"  default="tb3_2"/>
<arg name="3_tb3"  default="tb3_3"/>
<arg name="4_tb3"  default="tb3_4"/>
<arg name="5_tb3"  default="tb3_5"/>
<arg name="6_tb3"  default="tb3_6"/>
<arg name="7_tb3"  default="tb3_7"/>

<arg name="0_tb3_x_pos" default="-7.0"/>
<arg name="0_tb3_y_pos" default=" 3.0"/>
<arg name="0_tb3_z_pos" default=" 0.0"/>
<arg name="0_tb3_yaw"   default=" 1.57"/>

<arg name="1_tb3_x_pos" default="-1.0"/>
<arg name="1_tb3_y_pos" default=" 1.0"/>
<arg name="1_tb3_z_pos" default=" 0.0"/>
<arg name="1_tb3_yaw"   default=" 1.57"/>

<arg name="2_tb3_x_pos" default=" 4.0"/>
<arg name="2_tb3_y_pos" default=" 4.0"/>
<arg name="2_tb3_z_pos" default=" 0.0"/>
<arg name="2_tb3_yaw"   default=" 1.57"/>

<arg name="3_tb3_x_pos" default=" 9.0"/>
<arg name="3_tb3_y_pos" default=" 3.0"/>
<arg name="3_tb3_z_pos" default=" 0.0"/>
<arg name="3_tb3_yaw"   default=" 1.57"/>

<arg name="4_tb3_x_pos" default="-7.0"/>
<arg name="4_tb3_y_pos" default="-3.0"/>
<arg name="4_tb3_z_pos" default=" 0.0"/>
<arg name="4_tb3_yaw"   default=" 1.57"/>

<arg name="5_tb3_x_pos" default="-1.0"/>
<arg name="5_tb3_y_pos" default="-1.0"/>
<arg name="5_tb3_z_pos" default=" 0.0"/>
<arg name="5_tb3_yaw"   default=" 1.57"/>

<arg name="6_tb3_x_pos" default=" 4.0"/>
<arg name="6_tb3_y_pos" default="-4.0"/>
<arg name="6_tb3_z_pos" default=" 0.0"/>
<arg name="6_tb3_yaw"   default=" 1.57"/>

<arg name="7_tb3_x_pos" default=" 9.0"/>
<arg name="7_tb3_y_pos" default="-3.0"/>
<arg name="7_tb3_z_pos" default=" 0.0"/>
<arg name="7_tb3_yaw"   default=" 1.57"/>
```

![multi-robot](https://github.com/auto-swarm/auto-swarm/blob/new/src/simulation/turtlebot3_gazebo/assets/multi_robot.png)

issue:

```
[ WARN] [1652681145.557955870, 389.877000000]: TF_REPEATED_DATA ignoring data with redundant timestamp for frame base_footprint at time 389.876000 according to authority unknown_publisher
```

rebuild and relaunch launch file



# Deployment

* GPU/CUDA
* TensoRT

# How to make contributions

## create a new application package

```
catkin_create_pkg mpc roscpp rospy std_msgs
```

# Acknowledgements
