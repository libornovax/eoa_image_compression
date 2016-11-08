#include "OnePointCrossover.h"

#include <iostream>
#include "components/utils.h"
#include "components/RGen.h"
#include "shapes/Circle.h"
#include "components/Chromozome.h"
#include "components/Config.h"


namespace eic {


OnePointCrossover::OnePointCrossover (const cv::Size &image_size, const std::shared_ptr<Chromozome> &x)
    : _image_size(image_size),
      _x(x)
{
}


void OnePointCrossover::visit (Chromozome &chromozome)
{
    // Select a random position in the image
    std::uniform_int_distribution<int> distx(0, this->_image_size.width);
    std::uniform_int_distribution<int> disty(0, this->_image_size.height);

    cv::Point position(distx(RGen::mt()), disty(RGen::mt()));


    // Find all shapes in _x and chromozome that contain this position
    std::vector<int> idxs_x  = this->_containingIdxs(position, *this->_x);
    std::vector<int> idxs_ch = this->_containingIdxs(position, chromozome);


    // Exchange those shapes (or parts of them)
    for (int i = 0; i < idxs_x.size() && i < idxs_ch.size(); ++i)
    {
        chromozome[idxs_ch[i]] = this->_x->operator [](idxs_x[i]);
    }
}


void OnePointCrossover::visit (Circle &circle)
{

}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

std::vector<int> OnePointCrossover::_containingIdxs (const cv::Point &p, const Chromozome &chromozome)
{
    std::vector<int> containing_idxs;

    for (int i = 0; i < chromozome.size(); ++i)
    {
        if (chromozome[i]->contains(p))
        {
            containing_idxs.push_back(i);
        }
    }

    return containing_idxs;
}


}

