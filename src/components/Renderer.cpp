#include "Renderer.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "Chromozome.h"
#include "shapes/Circle.h"


namespace eic {

namespace {

    void drawLine (cv::Mat &image, int row, int x1, int x2, int color, double alpha)
    {
        int xmin = std::max(0, x1);
        int xmax = std::min(image.cols-1, x2);

        for (int j = xmin; j <= xmax; ++j)
        {
            image.at<uchar>(row, j) = (1-alpha)*image.at<uchar>(row, j) + alpha*color;
        }
    }


    void renderCircle (cv::Mat &image, const cv::Point &center, int radius, int color, double alpha)
    {
        int x_prev = 9999999;
        int x = radius;
        int y = 0;
        int err = 0;

        while (x >= y)
        {
            if (x != y)
            {
                if (center.y+y >= 0 && center.y+y < image.rows)
                {
                    // This row is inside of the image
                    drawLine(image, center.y + y, center.x - x, center.x + x, color, alpha);
                }
                if (y != 0 && center.y-y >= 0 && center.y-y < image.rows)
                {
                    // This row is inside of the image
                    drawLine(image, center.y - y, center.x - x, center.x + x, color, alpha);
                }
            }
            if (x_prev != x)
            {
                if (center.y+x >= 0 && center.y+x < image.rows)
                {
                    // This row is inside of the image
                    drawLine(image, center.y + x, center.x - y, center.x + y, color, alpha);
                }
                if (center.y-x >= 0 && center.y-x < image.rows)
                {
                    // This row is inside of the image
                    drawLine(image, center.y - x, center.x - y, center.x + y, color, alpha);
                }
            }

            x_prev = x;
            err += 1 + 2*y;
            y += 1;
            if (2*(err-x) + 1 > 0)
            {
                x -= 1;
                err += 1 - 2*x;
            }
        }
    }

}


Renderer::Renderer (const cv::Size &image_size)
    : _image_size(image_size)
{
    // The image has 3 channels - RGB
    this->_channels.resize(3);
    this->_reset();
}


const std::vector<cv::Mat> Renderer::render (Chromozome &ch)
{
    // Triger rendering of the chromozome -> visit it
    ch.accept(*this);

    return this->_channels;
}


void Renderer::visit (Chromozome &chromozome)
{
    // Reset all channels to 0 and the correct size
    this->_reset();

    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        // Render each shape
        chromozome[i]->accept(*this);
    }
}


void Renderer::visit (Circle &circle)
{
    double alpha = double(circle.getA()) / 100.0;

    renderCircle(this->_channels[0], circle.getCenter(), circle.getRadius(), circle.getR(), alpha);
    renderCircle(this->_channels[1], circle.getCenter(), circle.getRadius(), circle.getG(), alpha);
    renderCircle(this->_channels[2], circle.getCenter(), circle.getRadius(), circle.getB(), alpha);
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

