#include "CameraCalibrator.h"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>

// open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(const std::vector<std::string> &file_list, cv::Size &board_size)
{
    // the points on the chessboard
    std::vector<cv::Point2f> img_corners;
    std::vector<cv::Point3f> object_corners;
    // 3D scene points:
    // initialize the chesboard corners in the chessboard reference frame
    // the corners are at 3D location (x, y, z) = (i, j, 0)
    for (int i = 0; i < board_size.height; i++)
    {
        for (int j = 0; j < board_size.width; j++)
        {
            object_corners.push_back(cv::Point3f(i, j, 0.0f));
        }
    }
    // 2D image points to contain chessboard image
    cv::Mat img;
    int successes = 0;
    // for all viewpoints
    for (int i = 0; i < file_list.size(); i++)
    {
        // open image
        img = cv::imread(file_list[i], 0);
        // get the chessboard corners
        bool found = cv::findChessboardCorners(img, board_size, img_corners);
        // get subpixel accuracy on the corners
        // _, _, max number of iterations, min accuracy
        cv::cornerSubPix(img, img_corners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
        // if we have a good board add to our data
        if (img_corners.size() == board_size.area())
        {
            // add image and scene points from one view
            addPoints(img_corners, object_corners);
            successes++;
        }
        // draw the corners
        cv::drawChessboardCorners(img, board_size, img_corners, found);
        cv::imshow("Corners", img);
        cv::waitKey(100);
    }
    return successes;
}

double CameraCalibrator::calibrate(cv::Size &img_size)
{
    // undisorted must be reinitialized
    bool must_init_undisort = true;
    // start calibration
    // 3D points, image points, image size, output camera matrix, output distortion matrix, Rs, Ts, set options
    return calibrateCamera(object_points, img_points, img_size, camera_matrix, dist_coeffs, rvecs, tvecs, flag);
}

cv::Vec3d CameraCalibrator::triangulate(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2)
{
    // system of equations assuming image=[u, v] and X=[x, y, z, 1]
    // from u(p3.X)= p1.X and v(p3.X)=p2.X
    cv::Matx43d A(u1(0) * p1.at<double>(2, 0) - p1.at<double>(0, 0),
                  u1(0) * p1.at<double>(2, 1) - p1.at<double>(0, 1),
                  u1(0) * p1.at<double>(2, 2) - p1.at<double>(0, 2),
                  u1(1) * p1.at<double>(2, 0) - p1.at<double>(1, 0),
                  u1(1) * p1.at<double>(2, 1) - p1.at<double>(1, 1),
                  u1(1) * p1.at<double>(2, 2) - p1.at<double>(1, 2),
                  u2(0) * p2.at<double>(2, 0) - p2.at<double>(0, 0),
                  u2(0) * p2.at<double>(2, 1) - p2.at<double>(0, 1),
                  u2(0) * p2.at<double>(2, 2) - p2.at<double>(0, 2),
                  u2(1) * p2.at<double>(2, 0) - p2.at<double>(1, 0),
                  u2(1) * p2.at<double>(2, 1) - p2.at<double>(1, 1),
                  u2(1) * p2.at<double>(2, 2) - p2.at<double>(1, 2));

    cv::Matx41d B(p1.at<double>(0, 3) - u1(0) * p1.at<double>(2, 3),
                  p1.at<double>(1, 3) - u1(1) * p1.at<double>(2, 3),
                  p2.at<double>(0, 3) - u2(0) * p2.at<double>(2, 3),
                  p2.at<double>(1, 3) - u2(1) * p2.at<double>(2, 3));

    // X contains the 3D coordinate of the reconstructed point
    cv::Vec3d X;
    // solve AX=B
    cv::solve(A, B, X, cv::DECOMP_SVD);
    return X;
}

void showImg(cv::Mat image, int wait = 0, std::string win_name = "Image")
{
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(win_name, image);
    cv::waitKey(wait * 1000);
}

bool checkImgData(cv::Mat img, std::string path)
{
    if (!img.data)
    {
        std::cout << "Could not open file:" << std::endl;
        std::cout << path << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    cv::Mat img_l, img_r, img_concat, gray_l, gray_r;
    std::string project_path = "/Users/acano/Developer/Image_Processing/3D_Reconstruction";
    std::string data_path = project_path + "/data/cones";
    std::string img_path_left = data_path + "/left.png";
    std::string img_path_right = data_path + "/right.png";
    img_l = cv::imread(img_path_left, cv::IMREAD_COLOR);
    img_r = cv::imread(img_path_right, cv::IMREAD_COLOR);
    bool file_exists;
    file_exists = checkImgData(img_l, img_path_left);
    if (!file_exists)
    {
        return -1;
    }
    file_exists = checkImgData(img_r, img_path_right);
    if (!file_exists)
    {
        return -1;
    }
    cv::hconcat(img_l, img_r, img_concat);
    // convert to gray
    cv::cvtColor(img_l, gray_l, cv::COLOR_BGR2BGRA);
    cv::cvtColor(img_r, gray_r, cv::COLOR_BGR2GRAY);
    // define keypoints vector
    std::vector<cv::KeyPoint> kp_l, kp_r;
    // feature detector
    cv::Ptr<cv::Feature2D> ptr_feature_2d = cv::xfeatures2d::SURF::create(74);
    // keypint detection
    ptr_feature_2d->detect(img_l, kp_l);
    ptr_feature_2d->detect(img_r, kp_r);
    // extract descriptor
    cv::Mat des_l, des_r;
    ptr_feature_2d->compute(img_l, kp_l, des_l);
    ptr_feature_2d->compute(img_r, kp_r, des_r);
    // construction of the matcher
    cv::BFMatcher matcher(cv::NORM_L2);
    // match the two image descriptors
    std::vector<cv::DMatch> out_matches;
    matcher.match(des_l, des_r, out_matches);
    // convert keypoints into Points2f
    std::vector<cv::Point2d> pts_l, pts_r;
    for (std::vector<cv::DMatch>::const_iterator it = out_matches.begin(); it != out_matches.end(); ++it)
    {
        // get the position of keypoints
        pts_l.push_back(kp_l[it->queryIdx].pt);
        // get the position of right keypoints
        pts_r.push_back(kp_r[it->trainIdx].pt);
    }
    std::vector<uchar> inliers(pts_l.size(), 0);
    // args (matching points, match status (inlier or outlier), RANSAC method, distance to epipolar line, confidence probability)
    cv::Mat fundamental = cv::findFundamentalMat(pts_l, pts_r, inliers, cv::FM_RANSAC, 1.0, 0.98);
    std::cout << fundamental << std::endl;
    // compute homographyc rectification
    cv::Mat h_1, h_2;
    cv::stereoRectifyUncalibrated(pts_l, pts_r, fundamental, img_l.size(), h_1, h_2);
    // rectify images through wraping
    cv::Mat rectified_l, rectified_r;
    cv::warpPerspective(img_l, rectified_l, h_1, img_l.size());
    cv::warpPerspective(img_r, rectified_r, h_2, img_r.size());
    // compute disparity
    cv::Mat disparity;
    cv::Ptr<cv::StereoMatcher> p_stereo = cv::StereoSGBM::create(0, 32, 5);
    p_stereo->compute(rectified_l, rectified_r, disparity);
    cv::imwrite(data_path + "/disparity.jpg", disparity);
    // showImg(disparity);
    // essential matrix
    std::vector<cv::Mat> rvecs, tvecs;

    return 0;
}