# Quick Start within 3 Minutes 
Compiling tests passed on ubuntu **16.04, 18.04, and 20.04** with ros installed.
You can just execute the following commands one by one.
```
sudo apt-get install libarmadillo-dev
git clone https://github.com/hadleyhzy34/auto-swarm.git
cd auto-swarm
catkin_make -j1
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

### Copy install files into specific directory
```
cd CUSTOM_PATH/Ipopt-3.12.8/build
sudo cp -a include/* /usr/include/.
sudo cp -a lib/* /usr/lib/.
```


# Compilation

```
catkin_make -j$(nproc)
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

* Pure Pursuit
* PID
* LQR + PID
* MPC + IPOPT
* MPC + Casadi

# Simulation

# Deployment

* GPU/CUDA
* TensoRT

# How to make contributions

## create a new application package

```
catkin_create_pkg mpc roscpp rospy std_msgs
```

# Acknowledgements
