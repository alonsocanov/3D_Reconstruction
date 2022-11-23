#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <iostream>

void showImg(cv::Mat image, int wait = 0, std::string win_name = "Image");

bool checkImgData(cv::Mat img, std::string path);

#endif