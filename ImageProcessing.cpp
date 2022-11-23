#include "ImageProcessing.hpp"

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