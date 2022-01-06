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

    //for testing
    pcl::PointXYZ point1(10,2,3);
    Eigen::Vector4f eigen_point = point1.getVector4fMap();
    std::cout<<"vector point is: "<<eigen_point<<std::endl;
    
    // Create the PFH estimation class, and pass the input dataset+normals to it
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    pfh.setInputCloud (cloud);
    pfh.setInputNormals (normals);
    // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the PFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
    pfh.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    pfh.setRadiusSearch (0.05);

    // Compute the features
    pfh.compute (*pfhs);

    // pfhs->size () should have the same size as the input cloud->size ()*
    
    
    //visualize cloud with normals
    //visualCloudNormals(cloud,normals);
    
    //define plotter
    pcl::visualization::PCLPlotter *plotter = new pcl::visualization::PCLPlotter ("My Plotter");
	//attr
	plotter->setShowLegend (true);
	std::cout<<pcl::getFieldsList<pcl::PFHSignature125>(*pfhs);

	//visualize, pfhs->size: number of cloud points, 125->5*5*5, 5 bins for each dimension
    //it prints out histogram of each cloud point for each one of 125 bins
	for (int m=0; m<pfhs->size();m++)
	{
        std::cout<<"current point index is: : "<<m<<" "<<pfhs->size()<<std::endl;
		plotter->addFeatureHistogram<pcl::PFHSignature125>(*pfhs, "pfh", m, std::to_string(m)/*"one_fpfh"*/);
		plotter->setWindowSize(800, 600);
		plotter->spinOnce(1000);
	}
    while(!plotter->wasStopped())
    {
    }
	plotter->clearPlots();
    return 0;
}
