#include "Chromozome.h"

#include <chrono>
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
    Chromozome ch;
    for (int i = 0; i < Config::getParams().chromozome_length; ++i)
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
#ifdef MEASURE_TIME
    auto start = std::chrono::high_resolution_clock::now();
#endif
    const std::vector<cv::Mat> channels = renderer.render(*this);
#ifdef MEASURE_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Rendering time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;
#endif


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


void Chromozome::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


}
