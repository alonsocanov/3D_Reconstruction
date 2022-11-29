#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include "CameraCalibrator.hpp"

int main()
{

    std::cout << "Camera Calibration" << std::endl;

    const std::vector<std::string> files = {"boards/1.jpg", "boards/2.jpg", "boards/3.jpg",
                                            "boards/4.jpg", "boards/5.jpg", "boards/6.jpg", "boards/7.jpg",
                                            "boards/8.jpg", "boards/9.jpg", "boards/10.jpg", "boards/11.jpg",
                                            "boards/12.jpg", "boards/13.jpg", "boards/14.jpg", "boards/15.jpg",
                                            "boards/16.jpg", "boards/17.jpg", "boards/18.jpg", "boards/19.jpg",
                                            "boards/20.jpg", "boards/21.jpg", "boards/22.jpg", "boards/23.jpg",
                                            "boards/24.jpg", "boards/25.jpg"};
    cv::Size board_size(7, 7);
    // initialzing camera calibrator object / class
    CameraCalibrator cal = CameraCalibrator(1, false);
    int success = cal.addChessboardPoints(files, board_size);
    const int MIN_CHESSBOARD_IMGS = 10;
    if (success < MIN_CHESSBOARD_IMGS)
    {
        std::cout << "Unable to find at least " << MIN_CHESSBOARD_IMGS << " good chessboard images" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(files[0]);

    cv::Size img_size = img.size();
    cv::Mat = cameraMatrix = cal.calibrate(img_size);

    std::cout << "Calibration matrix: " << std::endl;

    std::cout << cameraMatrix << std::endl;

    cv::Mat image1 = cv::imread("imR.png");
    cv::Mat image2 = cv::imread("imL.png");

    // vector of keypoints and descriptors
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Construction of the SIFT feature detector
    cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SURF::create(10000);

    // Detection of the SIFT features and associated descriptors
    ptrFeature2D->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    ptrFeature2D->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Match the two image descriptors
    // Construction of the matcher with crosscheck
    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat matchImage;

    cv::namedWindow("img1");
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, matchImage, cv::Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("matches.jpg", matchImage);

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;

    for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
    {
        // Get the position of left keypoints
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x, y));
        // Get the position of right keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x, y));
    }

    // Find the essential between image 1 and image 2
    cv::Mat inliers;
    cv::Mat essential = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC, 0.9, 1.0, inliers);

    std::cout << essential << std::endl;

    // recover relative camera pose from essential matrix
    cv::Mat rotation, translation;
    cv::recoverPose(essential, points1, points2, cameraMatrix, rotation, translation, inliers);
    std::cout << rotation << std::endl;
    std::cout << translation << std::endl;

    // compose projection matrix from R,T
    cv::Mat projection2(3, 4, cv::CV_64F); // the 3x4 projection matrix
    rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(projection2.colRange(3, 4));
    // compose generic projection matrix
    cv::Mat projection1(3, 4, CV_64F, 0.); // the 3x4 projection matrix
    cv::Mat diag(cv::Mat::eye(3, 3, cv::CV_64F));
    diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
    // to contain the inliers
    std::vector<cv::Vec2d> inlierPts1;
    std::vector<cv::Vec2d> inlierPts2;
    // create inliers input point vector for triangulation
    int j = 0;
    for (int i = 0; i < inliers.rows; i++)
    {
        if (inliers.at<uchar>(i))
        {
            inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
            inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
        }
    }
    // undistort and normalize the image points
    std::vector<cv::Vec2d> points1u;
    cv::undistortPoints(inlierPts1, points1u, cameraMatrix, cal.distCoeffs);
    std::vector<cv::Vec2d> points2u;
    cv::undistortPoints(inlierPts2, points2u, cameraMatrix, cal.distCoeffs);

    // Triangulation
    std::vector<cv::Vec3d> points3D;
    cal.triangulate(projection1, projection2, points1u, points2u, points3D);

    std::cout << "3D points :" << points3D.size() << std::endl;
    // creating a Viz window
    cv::viz::Viz3d window;

    // Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", cv::viz::WCoordinateSystem());

    window.setBackgroundColor(cv::viz::Color::black());

    // Displaying the 3D points in green
    window.showWidget("points", cv::viz::WCloud(points3D, cv::viz::Color::green()));
    window.spin();
}