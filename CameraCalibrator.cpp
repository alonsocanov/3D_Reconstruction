#include "CameraCalibrator.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

CameraCalibrator::CameraCalibrator(int flag = 0, const bool mustInitUndistort = false)
{
    this->flag = flag;
    this->mustInitUndistort = mustInitUndistort;
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
// open chessboard images and extract corner points
int CameraCalibrator::addChessboardPoints(const std::vector<std::string> &file_list, cv::Size &board_size)
{
    // // the points on the chessboard
    std::vector<cv::Point3f> objectCorners;
    std::vector<cv::Point2f> imageCorners;
    // 3D scene points:
    // initialize the chesboard corners in the chessboard reference frame
    // the corners are at 3D location (x, y, z) = (i, j, 0)
    for (int i = 0; i < board_size.height; i++)
    {
        for (int j = 0; j < board_size.width; j++)
        {
            objectCorners.push_back(cv::Point3f(i, j, 0.0f));
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
        bool found = cv::findChessboardCorners(img, board_size, imageCorners);
        // get subpixel accuracy on the corners
        // _, _, max number of iterations, min accuracy
        cv::cornerSubPix(img, imageCorners, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.1));
        // if we have a good board add to our data
        if (imageCorners.size() == board_size.area())
        {
            // add image and scene points from one view
            addPoints(imageCorners, objectCorners);
            successes++;
        }
        // draw the corners
        cv::drawChessboardCorners(img, board_size, imagePoints, found);
        cv::imshow("Corners", img);
        cv::waitKey(100);
    }
    return successes;
}
// Add scene points and corresponding image points
void CameraCalibrator::addPoints(const std::vector<cv::Point2f> &imageCorners, const std::vector<cv::Point3f> &objectCorners)
{

    // 2D image points from one view
    imagePoints.push_back(imageCorners);
    // corresponding 3D scene points
    objectPoints.push_back(objectCorners);
}

double CameraCalibrator::calibrate(cv::Size &img_size)
{
    // undisorted must be reinitialized
    bool must_init_undisort = true;
    // start calibration
    // 3D points, image points, image size, output camera matrix, output distortion matrix, Rs, Ts, set options
    return calibrateCamera(this->objectPoints, this->imagePoints, img_size, cameraMatrix, distCoeffs, this->rvecs, this->tvecs, flag);
}

void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false)
{
    bool radial8CoeffEnabled = radial8CoeffEnabled;
    bool tangentialParamEnabled = tangentialParamEnabled;
}

cv::Mat CameraCalibrator::getCameraMatrix()
{
    return this->cameraMatrix;
}

cv::Mat CameraCalibrator::getDistCoeffs()
{
    return this->distCoeffs;
}
