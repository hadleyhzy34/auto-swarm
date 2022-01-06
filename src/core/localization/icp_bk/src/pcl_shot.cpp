#include <iostream>
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/histogram_visualizer.h>


/*---------------------read from txt file--------------------*/
void createCloudFromTxt(const std::string file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    //std::ifstream file(file_path.c_str());
    std::cout<<"current file path is: "<<file_path<<std::endl;
    std::ifstream file(file_path);
    std::string line;
    pcl::PointXYZ point;
    while(getline(file,line)){
        //std::cout<<"current line is: "<<line<<std::endl;
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
        //std::cout<<"current point is: "<<point.x<<" "<<point.y<<" "<<point.z<<std::endl;
        cloud->push_back(point);
    }
    file.close();
}

/*-----------------------pcl visualization-------------------*/
void visualCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer ("simple cloud viewer");
    viewer.showCloud(cloud);
    while(!viewer.wasStopped())
    {
    }
}

void visualCloudNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    pcl::visualization::PCLVisualizer viewer ("cloud and normal estimation viewer");
    viewer.setBackgroundColor(0.0,0.0,0.0);
    viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud,normals);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}

int main(int argc, char **argv) {
    std::cout << "Test PCL reading and visualization" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    createCloudFromTxt("/home/swarm/developments/point_cloud/modelnet40_normal_resampled/airplane/airplane_0001.txt",cloud);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	PCL_INFO ("Normal Estimation - Source\n");
	ne.setInputCloud (cloud);
    //create an empty kdtree representation
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree<pcl::PointXYZ> ());

	//ne.setSearchSurface (cloud_src);
	ne.setSearchMethod (tree_src);
	ne.setRadiusSearch (0.05);
	ne.compute (*normals);
    
    //declare SHOT descriptor, 2*2*8*11 bins
    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors (new pcl::PointCloud<pcl::SHOT352>);

    //define shot features
    pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352, pcl::ReferenceFrame> shot;
    //shot.setRadiusSearch (descr_rad);
    shot.setInputCloud (cloud);
    shot.setInputNormals (normals);
    //shot.setSearchSurface(cloud);
    shot.setSearchMethod(tree_src);
    shot.setRadiusSearch(0.05);
    //shot.setKSearch(10);
    shot.compute (*descriptors);

    //compute point size
    std::cout<<"shot output point size is: "<<descriptors->size()<<std::endl;

    //define plotter
    pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter ("My Plotter");
	//attFr
	plotter->setShowLegend (true);
	std::cout<<pcl::getFieldsList<pcl::SHOT352>(*descriptors);

	//visualize, pfhs->size: number of cloud points, 125->5*5*5, 5 bins for each dimension
    //it prints out histogram of each cloud point for each one of 125 bins
	for (int m=0; m<descriptors->size();m++)
	{
        std::cout<<"current dimension index is: "<<m<<" "<<descriptors->size()<<std::endl;
        //check output of each query point
        std::cout<<"shot descriptors "<<descriptors->points[m]<<std::endl;
        float sum = 0.;
        for(int i=0;i<descriptors->points[m].descriptorSize();i++){
            sum += descriptors->points[m].descriptor[i]*descriptors->points[m].descriptor[i];
        }
        std::cout<<"total sum of descriptor is: "<<sum<<std::endl;
        std::cout<<"current point local reference frame is: "<<std::endl;
        std::cout<<descriptors->points[m].rf<<std::endl;
        std::cout<<"current descriptor is: "<<std::endl;
        std::cout<<descriptors->points[m].descriptor<<std::endl;
        plotter->addFeatureHistogram<pcl::SHOT352>(*descriptors, "shot", m, std::to_string(m));
		plotter->setWindowSize(800, 600);
		plotter->spinOnce(100000);
	}
    while(!plotter->wasStopped())
    {
    }
	plotter->clearPlots();
    return 0;
}
