/* ************************************************ ************************
	> File Name: src/slamBase.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
    > Implementation of slamBase.h
	> Created Time: 15:31:49, Saturday, July 18, 2015
************************************************** ********************* */

# include  "slamBase.h"

PointCloud::Ptr image2PointCloud (cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud:: Ptr  cloud ( new PointCloud );

    for ( int m = 0 ; m <depth. rows ; m++)
        for ( int n = 0 ; n <depth. cols ; n++)
        {
            // Get the value at (m,n) in the depth map
            ushort d = depth. ptr < ushort >(m)[n];
            // d may have no value, if so, skip this point
            if (d == 0 )
                continue ;
            // d has a value, add a point to the point cloud
            PointT p;

            // Calculate the space coordinates of this point
            p. z = double (d) / camera. scale ;
            p. x = (n-camera. cx ) * p. z / camera. fx ;
            p. y = (m-camera. cy ) * p. z / camera. fy ;
            
            // Get its color from rgb image
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
    cloud-> is_dense = false ;

    return cloud;
}

cv::Point3f point2dTo3d (cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p; // 3D point
    p. z = double (point. z ) / camera. scale ;
    p. x = (point. x -camera. cx ) * p. z / camera. fx ;
    p. y = (point. y -camera. cy ) * p. z / camera. fy ;
    return p;
}

// computeKeyPointsAndDesp extracts key points and feature descriptors at the same time
void computeKeyPointsAndDesp( FRAME& frame, string detector, string descriptor )
{
    cv::Ptr <cv::FeatureDetector> _detector;
    cv::Ptr <cv::DescriptorExtractor> _descriptor;

    if(detector == "ORB")  _detector = cv::ORB::create();
    if(descriptor == "ORB") _descriptor = cv::ORB::create();

    if (!_detector || !_descriptor)
    {
        cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
        return;
    }

    _detector-> detect (frame.rgb, frame.kp );   // extract key points
    _descriptor-> compute (frame.rgb, frame.kp, frame.desp );   // Calculate the descriptor


    return;
}

// estimateMotion calculates the motion between two frames
// Input: frame 1 and frame 2
// Output: rvec and tvec
RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    static ParameterReader pd;
    vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );
   
    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }

    cout<<"good matches: "<<goodMatches.size()<<endl;

    RESULT_OF_PNP result;
    result.inliers = 0;

    if(goodMatches.size() < 5) return result;

    // 3D point of the first frame
    vector<cv::Point3f> pts_obj;
    // The image point of the second frame
    vector< cv::Point2f > pts_img;

    // Camera internal parameters
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query is the first, train is the second
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // Be careful to get d! x is to the right and y is down, so y is the row and x is the column!
        ushort d = frame1.depth.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // Convert (u,v,d) to (x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
        {camera.fx, 0, camera.cx},
        {0, camera.fy, camera.cy},
        {0, 0, 1}
    };

    cout<<"solving pnp"<<endl;
    // Build the camera matrix
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // Solve pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );

    
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

    return result;
}

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
  
    // Convert the translation vector and rotation matrix into a transformation matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    return T;
}

// joinPointCloud 
// Input: the original point cloud, the new frame and its pose
// Output: the image after adding the new frame to the original frame
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d T, CAMERA_INTRINSIC_PARAMETERS& camera ) 
{
    PointCloud::Ptr newCloud = image2PointCloud( newFrame.rgb, newFrame.depth, camera );

    // Merge point clouds
    PointCloud::Ptr output (new PointCloud());
    pcl::transformPointCloud( *original, *output, T.matrix() );
    *newCloud += *output;

    // Voxel grid filtering and downsampling
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud( newCloud );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    return tmp;
}


