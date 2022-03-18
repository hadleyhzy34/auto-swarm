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
## ros
```
sudo apt-get install ros-<distro>-desktop-full ros-<distro>-roscpp ros-<distro>-rospy ros-<distro>-nav-msgs ros-<distro>-nav-msgs
```

## ros dependencies
go to top of the folder and execute this command

```
rosdep install --from-paths src --ignore-src -r -y
```

## vtk installation

```
git clone --recursive https://gitlab.kitware.com/vtk/vtk.git
mkdir build
cd build
cmake ..
make -j12
sudo make install
```

## [pcl installation](https://pcl.readthedocs.io/en/latest/compiling_pcl_posix.html#compiling-pcl-posix)

go to [Github](https://github.com/PointCloudLibrary/pcl/releases) link and download certain version of PCL, uncompress the tar-gzip archive

```
cd pcl-pcl-1.8 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
sudo make -j2 install
```

## gtsam installation

```
git clone https://bitbucket.org/gtborg/gtsam.git
mkdir build
cd build
cmake ..
sudo make install
```
## Compilation

```
catkin_make -j$(nproc)
```

### issues and discussion

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

### Imitation Learning

## Localization && Mapping

## Perception

## Sensor Fusion

## Global Planner

## Local Planner

## Controller

# Simulation

# Deployment

# Acknowledgements
