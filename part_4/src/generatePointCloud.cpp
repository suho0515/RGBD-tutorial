// C++ Standard Library
# include  <iostream>
# include  <string>
using  namespace  std ;

// OpenCV library
# include  <opencv2/core/core.hpp>
# include  <opencv2/highgui/highgui.hpp>

// PCL library
# include  <pcl/io/pcd_io.h>
# include  <pcl/point_types.h>

// Define point cloud type
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// Camera internal parameters
const  double camera_factor = 1000 ;
const  double camera_cx = 325.5 ;
const  double camera_cy = 253.5 ;
const  double camera_fx = 518.0 ;
const  double camera_fy = 519.0 ;

// Main function
int  main ( int argc, char ** argv)
{
    // Read ./data/rgb.png and ./data/depth.png and convert them to point cloud

    // Image matrix
    cv::Mat rgb, depth;
    // Use cv::imread() to read the image
    // API: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#cv2.imread
    rgb = cv::imread ( "./data/rgb.png");
    // rgb image is a color image of 8UC3
    // depth is a single-channel image of 16UC1, pay attention to the flags setting -1, which means reading the original data without any modification
    depth = cv::imread ( "./data/depth.png");

    // Point cloud variables
    // Use smart pointers to create an empty point cloud. This pointer will be automatically released when it runs out.
    PointCloud:: Ptr  cloud ( new PointCloud );
    // Traverse the depth map
    for ( int m = 0 ; m <depth. rows ; m++)
        for ( int n = 0 ; n <depth. cols ; n++)
        {
            // Get the value at (m,n) in the depth map
            ushort d = depth. ptr < ushort >(m)[n];
            // d may have no value, if so, skip this point
            if (d == 0 )
                continue ;
            // If d has a value, add a point to the point cloud
            PointT p;

            // Calculate the space coordinates of this point
            p. z = double (d) / camera_factor;
            p. x = (n-camera_cx) * p. z / camera_fx;
            p. y = (m-camera_cy) * p. z / camera_fy;
            
            // Get its color from the rgb image
            // rgb is a three-channel BGR format image, so get the colors in the following order
            p. b = rgb. ptr <uchar>(m)[n* 3 ];
            p. g = rgb. ptr <uchar>(m)[n* 3 + 1 ];
            p. r = rgb. ptr <uchar>(m)[n* 3 + 2 ];

            // Add p to the point cloud
            cloud-> points . push_back (p );
        }
    // Set and save point cloud
    cloud-> height = 1 ;
    cloud-> width = cloud-> points . size ();
    cout<< " point cloud size = " <<cloud-> points . size ()<<endl;
    cloud-> is_dense = false ;
    pcl::io::savePCDFile ("./pointcloud.pcd", *cloud );
    // Clear data and exit
    cloud-> points . clear ();
    cout<< " Point cloud saved. " <<endl;
    return  0 ;
}
