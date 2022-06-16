# Quick Start within 3 Minutes 
Compiling tests passed on ubuntu **16.04, 18.04, and 20.04** with ros installed.
You can just execute the following commands one by one.
```
sudo apt-get install libarmadillo-dev
git clone https://github.com/hadleyhzy34/auto-swarm.git
cd auto-swarm
catkin_make -j$(nproc)
source devel/setup.bash
```

# Environment
PCL --version 1.8

GTSAM --version 4

VTK --version 7.1.1

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

## vtk installation(optional)

```
git clone --recursive https://gitlab.kitware.com/vtk/vtk.git
mkdir build
cd build
cmake ..
make -j12
sudo make install
```

## [pcl installation](https://pcl.readthedocs.io/en/latest/compiling_pcl_posix.html#compiling-pcl-posix)(optional)

go to [Github](https://github.com/PointCloudLibrary/pcl/releases) link and download certain version of PCL, uncompress the tar-gzip archive

```
cd pcl-pcl-1.8 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
sudo make -j2 install
```

## gtsam installation(optional)

```
git clone https://bitbucket.org/gtborg/gtsam.git
mkdir build
cd build
cmake ..
sudo make install
```

## [Ipopt](https://coin-or.github.io/Ipopt/INSTALL.html) installation(optional)

### Install CPPAD & Fortran
```
sudo apt-get install cppad gfortran
```

### Get ipopt source code

```
wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.9.tgz
tar -zxvf Ipopt-3.12.9.tgz
```

### Step by step download the libraries

```
cd Ipopt-3.12.9/ThirdParty/Blas
 ./get.Blas
 cd ../Lapack
 ./get.Lapack
 cd ../Mumps
 ./get.Mumps
 cd ../Metis
 ./get.Metis
```

```
cd {$CUSTOM_PATH}/Ipopt-3.12.9
mkdir build  && cd build
../configure --prefix=/usr/local
make -j$(nproc)
sudo make install
```

* issue: mpi.h file or dir not found:
  Solution: make sure third party lib Mumps is downloaded and uncompressed properly

* issue: anaconda3/lib/libfontconfig.so.1: 'FT_Done_MM_Var' undefined reference collect2: error: ld returned 1 exit status
  Solution: 
    ```
    sudo rm /home/**/anaconda3/lib/libuuid.so.1
    sudo ln -s /lib/x86_64-linux-gnu/libuuid.so.1  /home/**/anaconda3/lib/libuuid.so.1
    ```

# Compilation

```
git clone https://github.com/hadleyhzy34/auto-swarm.git
cd auto-swarm
catkin_make -j$(nproc)
source devel/setup.bash
```

## issues and discussion

* issue1
    * issue: undefined reference to 'pcl::KdTreeFLANN'
    * solution:
    ```
    #include <pcl/search/impl/kdtree.hpp>
    #include <pcl/kdtree/impl/kdtree_flann.hpp>
    ```

* issue2
    * issue: error while loading shared libraries: libvtkglew-7.1.so.1: cannot open shared object file: No such file or directory
    * solution:
    ```
    sudo ldconfig -v
    ```
* issue3
    * issue: fatal error: multi_map_server/MultiOccupancyGrid.h: No such file or directory
    * solution:
    ```
    sudo apt-get install ros-melodic-multi-map-server
    ```


# Components && Applications

## end to end

### Reinforcement Learning

* on-policy
* off-policy

Features:
* [x]parallel multi-agent training
* [ ]ground truth map label
* [x]GPU supported training and inference
* [x]sensor synchronization mechanism
* [ ]baseline methods support
* [x]Gym-like Env wrapper


#### Gazebo based environment wrapper

1. conda environment

```
conda create -n ros_tb3 pyhton=3.6.9
conda activate ros_tb3
```

2. pip installation

```
pip install torch
```

### `Env()` API

#### Initializing Environment

```
from env.env import Env
env = Env(robot_namespace=None)
```

#### Standard methods

* Stepping
```
env.step(self, action: ActType) -> state:numpy.array, reward: float, done: bool
```

* Resetting
```
env.step(self) -> state:numpy.array
```
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

### [MPC](https://github.com/hadleyhzy34/auto-swarm/tree/main/src/core/controller/mpc_python_traj)(Casadi/python2)


# Simulation

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
