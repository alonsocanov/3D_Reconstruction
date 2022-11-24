#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
// #include "CameraCalibrator.hpp"
#include "ImageProcessing.hpp"

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

    return 0;
}