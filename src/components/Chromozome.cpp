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


std::shared_ptr<Chromozome> Chromozome::clone () const
{
    auto ch = std::make_shared<Chromozome>();

    for (auto &shape: this->_chromozome)
    {
        ch->_chromozome.push_back(shape->clone());
    }

    return ch;
}


std::shared_ptr<Chromozome> Chromozome::randomChromozome (const cv::Size &image_size)
{
    auto ch = std::make_shared<Chromozome>();
    for (int i = 0; i < Config::getParams().chromozome_length; ++i)
    {
        ch->addRandomShape(image_size);
    }
    return ch;
}


size_t Chromozome::size () const
{
    return this->_chromozome.size();
}


void Chromozome::addRandomShape (const cv::Size &image_size)
{
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


const std::shared_ptr<IShape>& Chromozome::operator[] (size_t i) const
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
        cv::subtract(target[i], channels[i], diff, cv::noArray(), CV_32FC1);
        cv::pow(diff, 2, diff);
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
    Renderer r(image_size);
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
