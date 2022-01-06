#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


/*---------------------read from txt file--------------------*/
void createCloudFromTxt(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    //std::ifstream file(file_path.c_str());
    std::cout<<"current file path is: "<<file_path<<std::endl;
    std::ifstream file(file_path);
    std::string line;
    pcl::PointXYZ point;
    while(getline(file,line)){
        std::cout<<"current line is: "<<line<<std::endl;
        std::stringstream ss(line);
        std::string x;
        std::getline(ss, x, ',');
        point.x = std::stof(x);
        std::string y;
        std::getline(ss, y, ',');
        point.y = std::stof(y);
        std::string z;
        std::getline(ss, z, ',');
        point.z = std::stof(z);
        std::cout<<"current point is: "<<point.x<<" "<<point.y<<" "<<point.z<<std::endl;
        cloud->push_back(point);
    }
    file.close();
}

/*-----------------------pcl visualization-------------------*/
void visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer ("simple cloud viewer");
    viewer.showCloud(cloud);
    while(!viewer.wasStopped())
    {
    }
}


int main(int argc, char **argv) {
    std::cout << "Test PCL reading and visualization" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    createCloudFromTxt("/home/swarm/developments/point_cloud/modelnet40_normal_resampled/airplane/airplane_0001.txt",cloud);
    visualization(cloud);
    return 0;
}
