#include "Renderer.h"

#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include "components/Chromozome.h"
#include "shapes/Circle.h"


namespace eic {

namespace {

    /**
     * @brief Adds the specified color value to the specified horizontal line segment
     * @param image Image (CV_32SC1) to be altered
     * @param row Row in the image, where the line is going to be drawn
     * @param x1 Left edge pixel of the line
     * @param x2 Right edge pixel of the line
     * @param color Number to be added to the current values of the pixels
     */
#ifdef RENDER_AVERAGE
    void drawLine (cv::Mat &image, int row, int x1, int x2, int color)
#else
    void drawLine (cv::Mat &image, int row, int x1, int x2, int alpha_color, double alpha)
#endif
    {
#ifdef RENDER_AVERAGE
        assert(image.type() == 4);
#else
        assert(image.type() == 0);
#endif

        // Clip the coordinates if they do not fit inside of the image
        const int xmin = std::max(0, x1);
        const int xmax = std::min(image.cols-1, x2);

#ifdef RENDER_AVERAGE
        int* ptr_row_x = image.ptr<int>() + row*image.cols + xmin;
        int* ptr_row_x_max = image.ptr<int>() + row*image.cols + xmax;
#else
        uchar* ptr_row_x = image.ptr<uchar>() + row*image.cols + xmin;
        uchar* ptr_row_x_max = image.ptr<uchar>() + row*image.cols + xmax;
#endif

        for (; ptr_row_x <= ptr_row_x_max; ptr_row_x++)
        {
#ifdef RENDER_AVERAGE
            *ptr_row_x += color;
#else
            *ptr_row_x = uchar((1-alpha)*(*ptr_row_x) + alpha_color);
#endif
        }
    }


    /**
     * @brief Adds the specified color value to the specified circle shape into the image
     * @param image Image (CV_32SC1) to be altered
     * @param center
     * @param radius
     * @param color Number to be added to the current values of the pixels
     */
#ifdef RENDER_AVERAGE
    void renderCircle (cv::Mat &image, const cv::Point &center, int radius, int color)
#else
    void renderCircle (cv::Mat &image, const cv::Point &center, int radius, int alpha_color, double alpha)
#endif
    {
        // The plotting of the circle is done with the "Midpoint circle algorithm" as described here:
        // https://en.wikipedia.org/wiki/Midpoint_circle_algorithm. There are some alternations to the
        // provided code to prevent a line being processed several times, which happens in the provided
        // implementation of the article

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
#ifdef RENDER_AVERAGE
                    drawLine(image, center.y + y, center.x - x, center.x + x, color);
#else
                    drawLine(image, center.y + y, center.x - x, center.x + x, alpha_color, alpha);
#endif
                }
                if (y != 0 && center.y-y >= 0 && center.y-y < image.rows)
                {
                    // This row is inside of the image
#ifdef RENDER_AVERAGE
                    drawLine(image, center.y - y, center.x - x, center.x + x, color);
#else
                    drawLine(image, center.y - y, center.x - x, center.x + x, alpha_color, alpha);
#endif
                }
            }
            // We only want to plot this line if the x coordinate changed
            if (x_prev != x)
            {
                if (center.y+x >= 0 && center.y+x < image.rows)
                {
                    // This row is inside of the image
#ifdef RENDER_AVERAGE
                    drawLine(image, center.y + x, center.x - y, center.x + y, color);
#else
                    drawLine(image, center.y + x, center.x - y, center.x + y, alpha_color, alpha);
#endif
                }
                if (center.y-x >= 0 && center.y-x < image.rows)
                {
                    // This row is inside of the image
#ifdef RENDER_AVERAGE
                    drawLine(image, center.y - x, center.x - y, center.x + y, color);
#else
                    drawLine(image, center.y - x, center.x - y, center.x + y, alpha_color, alpha);
#endif
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
#ifdef RENDER_AVERAGE
    // The image (cumulative sum image) has 4 channels - RGBA
    this->_channels.resize(4);
#else
    // The image has 3 channels - RGB
    this->_channels.resize(3);
#endif
}


const std::vector<cv::Mat> Renderer::render (Chromozome &ch)
{
    // Triger rendering of the chromozome -> visit it
    ch.accept(*this);

    return this->_getRenderedChannels();
}


void Renderer::visit (Chromozome &chromozome)
{
#ifdef RENDER_AVERAGE
    // The image is rendered as follows: We compute the weighted average for each pixel over all shapes in
    // the chromozome. Therefore, we first compute the weighted sums over all shapes for each pixel of the
    // image and in the end we divide them by the weight sum for each pixel.
#else
    // The image is rendered as follows: We add shapes one by one in the order in which they are in the
    // chromozome. This means that the final image depends on the order of the shapes in the chromozome.
    // The value, which is in the image is weighted by (1-alpha) and the new value by alpha.
#endif

    // Reset all channels to 0 and the correct size
    this->_reset();

    for (int i = chromozome.size()-1; i >= 0; --i)
    {
        // Render each shape
        chromozome[i]->accept(*this);
    }
}


void Renderer::visit (Circle &circle)
{
    double alpha = double(circle.getA()) / 100.0;

#ifdef RENDER_AVERAGE
    // Render (add) the values for each color channel
    renderCircle(this->_channels[0], circle.getCenter(), circle.getRadius(), alpha*circle.getR());
    renderCircle(this->_channels[1], circle.getCenter(), circle.getRadius(), alpha*circle.getG());
    renderCircle(this->_channels[2], circle.getCenter(), circle.getRadius(), alpha*circle.getB());
    // Render the sum of the alpha channel
    renderCircle(this->_channels[3], circle.getCenter(), circle.getRadius(), circle.getA());
#else
    renderCircle(this->_channels[0], circle.getCenter(), circle.getRadius(), alpha*circle.getR(), alpha);
    renderCircle(this->_channels[1], circle.getCenter(), circle.getRadius(), alpha*circle.getG(), alpha);
    renderCircle(this->_channels[2], circle.getCenter(), circle.getRadius(), alpha*circle.getB(), alpha);
#endif
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Renderer::_reset ()
{
    for (cv::Mat &channel: this->_channels)
    {
#ifdef RENDER_AVERAGE
        // Set the whole image to black (use 1 to prevent further division by 0)
        channel = cv::Mat(this->_image_size, CV_32SC1, cv::Scalar(1));
#else
        channel = cv::Mat(this->_image_size, CV_8UC1, cv::Scalar(0));
#endif
    }
}


std::vector<cv::Mat> Renderer::_getRenderedChannels ()
{
#ifdef RENDER_AVERAGE
    // Because the RGB values are weighted averages over all shapes in the image, we need to divide them
    // now by the total sum of alpha values, which are stored int the 4th channel
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < channels.size(); ++i)
    {
        cv::divide(this->_channels[i], this->_channels[3], channels[i], 100, CV_8UC1);
    }

    return channels;
#else
    return this->_channels;
#endif
}


}

