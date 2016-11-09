#include "OnePointCrossover.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
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

//    {
//        cv::Mat canvas(this->_image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_x.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(this->_x->operator [](idxs_x[i]));
//            cv::circle(canvas, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::Mat canvas2(this->_image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_ch.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(chromozome[idxs_ch[i]]);
//            cv::circle(canvas2, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::imshow("crossover x", canvas);
//        cv::imshow("crossover ch", canvas2);
//        std::cout << "Crossover size: " << idxs_x.size() << "  " << idxs_ch.size() << std::endl;
//        cv::waitKey();
//    }

    // Exchange those shapes (or parts of them)
    for (int i = 0; i < idxs_x.size() && i < idxs_ch.size(); ++i)
    {
        chromozome[idxs_ch[i]] = this->_x->operator [](idxs_x[i])->clone();
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

