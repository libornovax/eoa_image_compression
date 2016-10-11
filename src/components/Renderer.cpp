#include "Renderer.h"


namespace eic {


Renderer::Renderer (const cv::Size &size)
{
    // The image has 3 channels - RGB
    this->_channels.resize(3);

    for (cv::Mat &channel: this->_channels)
    {
        channel = cv::Mat(size, CV_8UC1, cv::Scalar(0));
    }
}


const std::vector<cv::Mat> Renderer::render (const Chromozome &ch)
{
    for (auto &shape: ch)
    {
        // Render each shape
        shape->accept(*this);
    }
}


void Renderer::visit (const Circle &circle)
{
    std::cout << "plotting circle" << std::endl;
}


}

