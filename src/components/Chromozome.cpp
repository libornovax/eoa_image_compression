#include "Chromozome.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "Renderer.h"
#include "Config.h"
#include "shapes/Circle.h"


namespace eic {


Chromozome::Chromozome()
    : _difference(DBL_MAX)
{

}


Chromozome Chromozome::clone () const
{
    Chromozome ch;

    for (auto &shape: this->_chromozome)
    {
        ch._chromozome.push_back(shape->clone());
    }

    return ch;
}


Chromozome Chromozome::randomChromozome (const cv::Size &image_size)
{
    // Length to which the chromozome is initialized (default is 5)
    int init_length = std::min(5, Config::getParams().chromozome_length);

    Chromozome ch;
    for (int i = 0; i < init_length; ++i)
    {
        ch.addRandomShape(image_size);
    }
    return ch;
}


size_t Chromozome::size () const
{
    return this->_chromozome.size();
}


void Chromozome::addRandomShape (const cv::Size &image_size)
{
    // All shapes in the current chromozome will be old ones
    for (auto &shape: this->_chromozome) shape->setOld();

    // Do not add anything if it is long enough
    if (this->_chromozome.size() >= Config::getParams().chromozome_length) return;

    switch (Config::getParams().shape_type) {
    case ShapeType::CIRCLE:
        this->_chromozome.push_back(Circle::randomCircle(image_size));
        break;
    default:
        std::cout << "ERROR: Unknown shape type " << int(Config::getParams().shape_type) << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
}


std::shared_ptr<IShape>& Chromozome::operator[] (size_t i)
{
    assert(i < this->_chromozome.size());

    return this->_chromozome[i];
}


double Chromozome::computeDifference (const std::vector<cv::Mat> &target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());

    // Render the image
    Renderer renderer(target[0].size());
    const std::vector<cv::Mat> channels = renderer.render(*this);


    // Compute pixel-wise difference
    this->_difference = 0;
    for (size_t i = 0; i < target.size(); ++i)
    {
        cv::Mat diff;
        cv::absdiff(target[i], channels[i], diff);
        cv::Scalar total = cv::sum(diff);
        this->_difference += total[0];
    }

    return this->_difference;
}


double Chromozome::getDifference () const
{
    return this->_difference;
}


cv::Mat Chromozome::asImage (const cv::Size &image_size)
{
    // Render the image represented by this chromozome
    eic::Renderer r(image_size);
    const std::vector<cv::Mat> channels = r.render(*this);

    // Merge the channels to a 3 channel cv::Mat
    cv::Mat image;
    cv::merge(channels, image);
    cv::cvtColor(image, image, CV_RGB2BGR);

    return image;
}


void Chromozome::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


}
