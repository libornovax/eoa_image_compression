#include "Renderer.h"

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

    for (auto &shape: ch)
    {
        // Render each shape
        shape->accept(*this);
    }

    return this->_channels;
}


void Renderer::visit (const Circle &circle)
{
    cv::circle(this->_channels[0], circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getR()), -1);
    cv::circle(this->_channels[1], circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getG()), -1);
    cv::circle(this->_channels[2], circle.getCenter(), circle.getRadius(), cv::Scalar(circle.getB()), -1);
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Renderer::_reset ()
{
    for (cv::Mat &channel: this->_channels)
    {
        // Set the whole image to black
        channel = cv::Mat(this->_image_size, CV_8UC1, cv::Scalar(0));
    }
}


}

