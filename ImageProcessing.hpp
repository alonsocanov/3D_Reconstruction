#ifndef IMAGEPROCESSING_HPP
#define IMAGEPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <iostream>

void showImg(cv::Mat image, int wait, std::string win_name);

bool checkImgData(cv::Mat img, std::string path);

#endif