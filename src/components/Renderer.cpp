#include "Renderer.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "shapes/Circle.h"


namespace eic {


Renderer::Renderer (const cv::Size &image_size)
    : _image_size(image_size)
{
    // The image has 3 channels - RGB
    this->_channels.resize(3);
    this->_reset();
}


const std::vector<cv::Mat> Renderer::render (const Chromozome &ch)
{
    // Reset all channels to 0 and the correct size
    this->_reset();

    for (auto &shape: ch.chromozome())
    {
        // Render each shape
        shape->accept(*this);
    }

    return this->_channels;
}


void Renderer::visit (const Circle &circle)
{
    double alpha = double(circle.getA()) / 100.0;

    cv::Mat overlayr;
    this->_channels[0].copyTo(overlayr);
    cv::circle(overlayr, circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getR()), -1);
    cv::addWeighted(overlayr, alpha, this->_channels[0], 1-alpha, 0, this->_channels[0]);

    cv::Mat overlayg;
    this->_channels[1].copyTo(overlayg);
    cv::circle(overlayg, circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getG()), -1);
    cv::addWeighted(overlayg, alpha, this->_channels[1], 1-alpha, 0, this->_channels[1]);

    cv::Mat overlayb;
    this->_channels[2].copyTo(overlayb);
    cv::circle(overlayb, circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getB()), -1);
    cv::addWeighted(overlayb, alpha, this->_channels[2], 1-alpha, 0, this->_channels[2]);
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Renderer::_reset ()
{
    for (cv::Mat &channel: this->_channels)
    {
        // Set the whole image to black
        channel = cv::Mat(this->_image_size, CV_8UC1, cv::Scalar(255));
    }
}


}

