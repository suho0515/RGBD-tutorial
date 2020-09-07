/* ************************************************ ************************
	> File Name: detectFeatures.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
    > Feature extraction and matching
	> Created Time: Saturday, July 18, 2015, 16:00:21
************************************************** ********************* */

# include <iostream>
# include  "slamBase.h"
using  namespace  std ;

// OpenCV feature detection module
# include  <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
# include  <opencv2/calib3d/calib3d.hpp>

int  main ( int argc, char ** argv)
{
    // Declare and read two rgb and depth maps from the data folder
    cv::Mat rgb1 = cv::imread("./data/rgb1.png" );
    cv::Mat rgb2 = cv::imread("./data/rgb2.png" );
    cv::Mat depth1 = cv::imread("./data/depth1.png");
    cv::Mat depth2 = cv::imread("./data/depth2.png");

    // Declare feature extractor and descriptor extractor
    cv:: Ptr <cv::FeatureDetector> detector;
    cv:: Ptr <cv::DescriptorExtractor> descriptor;

    // Build the extractor, both are ORB by default
    
    // If you use sift, surf, you must initialize the nonfree module before
    // cv::initModule_nonfree();
    // _detector = cv::FeatureDetector::create( "SIFT" );
    // _descriptor = cv::DescriptorExtractor::create( "SIFT" );
    
    detector = cv::ORB::create();
    descriptor = cv::ORB::create();

    vector< cv::KeyPoint> kp1, kp2; // Key point
    detector-> detect (rgb1, kp1 );   // extract key points
    detector-> detect (rgb2, kp2 );

    cout<< " Key points of two images: " <<kp1. size ()<< " , " <<kp2. size ()<<endl;
    
    // Visualization, display key points
    cv::Mat imgShow;
    cv::drawKeypoints (rgb1, kp1, imgShow, cv::Scalar::all (- 1 ), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow ( "keypoints" , imgShow );
    cv::imwrite ( "./data/keypoints.png" , imgShow );
    cv::waitKey ( 0 ); // Pause and wait for a button
   
    // Calculate the descriptor
    cv::Mat desp1, desp2;
    descriptor-> compute (rgb1, kp1, desp1 );
    descriptor-> compute (rgb2, kp2, desp2 );

    // Match descriptor
    vector< cv::DMatch> matches; 
    cv::BFMatcher matcher;
    matcher. match (desp1, desp2, matches );
    cout<< " Find total " <<matches. size ()<< " matches. " <<endl;

    // Visualization: display matching features
    cv::Mat imgMatches;
    cv::drawMatches (rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow ( "matches" , imgMatches );
    cv::imwrite ( "./data/matches.png" , imgMatches );
    cv::waitKey ( 0 );

    // Filter matches and remove those that are too far away
    // The criterion used here is to remove matches greater than four times the minimum distance
    vector< cv::DMatch> goodMatches;
    double minDis = 9999 ;
    for ( size_t i = 0 ; i<matches. size (); i++)
    {
        if (matches[i]. distance <minDis)
            minDis = matches[i]. distance ;
    }
    cout<< " min dis = " <<minDis<<endl;

    for ( size_t i = 0 ; i<matches. size (); i++)
    {
        if (matches[i]. distance < 10 *minDis)
            goodMatches. push_back (matches[i] );
    }

    // Display good matches
    cout<< " good matches= " <<goodMatches. size ()<<endl;
    cv::drawMatches (rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow ( "good matches" , imgMatches );
    cv::imwrite ( "./data/good_matches.png" , imgMatches );
    cv::waitKey ( 0 );

    // Calculate the motion relationship between images
    // Key function: cv::solvePnPRansac()
    // Prepare the necessary parameters for calling this function
    
    // 3D point of the first frame
    vector<cv::Point3f> pts_obj;
    // The image point of the second frame
    vector< cv::Point2f> pts_img;

    // Camera internal parameters
    CAMERA_INTRINSIC_PARAMETERS C;
    C. cx = 325.5 ;
    C. cy = 253.5 ;
    C. fx = 518.0 ;
    C. fy = 519.0 ;
    C. scale = 1000.0 ;

    for ( size_t i = 0 ; i<goodMatches. size (); i++)
    {
        // query is the first, train is the second
        cv::Point2f p = kp1[goodMatches[i]. queryIdx ]. pt ;
        // Be careful to get d! x is to the right and y is down, so y is the row and x is the column!
        ushort d = depth1.ptr<ushort>(int(p.y))[int(p.x)];
        if (d == 0 )
            continue ;
        pts_img. push_back ( cv::Point2f ( kp2 [goodMatches[i]. trainIdx ]. pt ) );

        // Convert (u,v,d) to (x,y,z)
        cv::Point3f pt (p. x , p. y , d );
        cv::Point3f pd = point2dTo3d (pt, C );
        pts_obj. push_back (pd );
    }

    double camera_matrix_data[ 3 ][ 3 ] = {
        {C. fx , 0 , C. cx },
        { 0 , C. fy , C. cy },
        { 0 , 0 , 1 }
    };

    // Build the camera matrix
    cv::Mat cameraMatrix ( 3 , 3 , CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // Solve pnp
    cv::solvePnPRansac (pts_obj, pts_img, cameraMatrix, cv::Mat (), rvec, tvec, false, 100, 1.0, 0.99, inliers );

    cout<< " inliers: " <<inliers. rows <<endl;
    cout<< " R= " <<rvec<<endl;
    cout<< " t= " <<tvec<<endl;

    // Draw inliers matching
    vector< cv::DMatch> matchesShow;
    for ( size_t i = 0 ; i<inliers. rows ; i++)
    {
        matchesShow. push_back (goodMatches[inliers. ptr < int >(i)[ 0 ]] );    
    }
    cv::drawMatches (rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow ( "inlier matches" , imgMatches );
    cv::imwrite ( "./data/inliers.png" , imgMatches );
    cv::waitKey ( 0 );

    return  0 ;
}
