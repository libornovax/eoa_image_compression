#include "Chromozome.h"

#include "Renderer.h"


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


std::vector<std::shared_ptr<IShape>>& Chromozome::chromozome ()
{
    return this->_chromozome;
}


const std::vector<std::shared_ptr<IShape>>& Chromozome::chromozome () const
{
    return this->_chromozome;
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
    for (int i = 0; i < target.size(); ++i)
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
