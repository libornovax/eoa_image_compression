#include "DifferentialCrossover.h"

#include <iostream>
#include "components/utils.h"
#include "components/RGen.h"
#include "shapes/Circle.h"
#include "components/Chromozome.h"
#include "components/Config.h"


namespace eic {


DifferentialCrossover::DifferentialCrossover (const cv::Size &image_size,
                                              const std::shared_ptr<Chromozome> &x1,
                                              const std::shared_ptr<Chromozome> &x2,
                                              const std::shared_ptr<Chromozome> &x3)
    : _image_size(image_size),
      _x1(x1),
      _x2(x2),
      _x3(x3)
{
}


void DifferentialCrossover::visit (Chromozome &chromozome)
{
    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        if (utils::makeMutation(Config::getParams().differential_crossover.shape_crossover_prob))
        {
            // Invoke crossover of the shape
            chromozome[i]->accept(*this);
        }
    }
}


void DifferentialCrossover::visit (Circle &circle)
{
    // Find the closest circles in all 3 chromozomes
    auto c1 = this->_findClosestCircle(circle, this->_x1);
    auto c2 = this->_findClosestCircle(circle, this->_x2);
    auto c3 = this->_findClosestCircle(circle, this->_x3);

    // Perform the differential crossover
    // Select one of the features of the circle, which will undergo crossover (only one feature at the time
    // will undergo crossover)
    std::uniform_int_distribution<int> distf(0, 5);  // RGBArc
    int crossover_feature = distf(RGen::mt());

    switch (crossover_feature)
    {
    case 0:
        // Channel R
        {
            circle._r = c1->_r + (c2->_r-c3->_r);
            circle._r = utils::clip(circle._r, 0, 255);
        }
        break;
    case 1:
        // Channel G
        {
            circle._g = c1->_g + (c2->_g-c3->_g);
            circle._g = utils::clip(circle._g, 0, 255);
        }
        break;
    case 2:
        // Channel B
        {
            circle._b = c1->_b + (c2->_b-c3->_b);
            circle._b = utils::clip(circle._b, 0, 255);
        }
        break;
    case 3:
        // Channel alpha
        {
            circle._a = c1->_a + (c2->_a-c3->_a);
            circle._a = utils::clip(circle._a, 30, 60);
        }
        break;
    case 4:
        // Radius
        {
            circle._radius = c1->_radius + (c2->_radius-c3->_radius);
            circle._radius = utils::clip(circle._radius, 2, 10000); // Must be positive
        }
        break;
    case 5:
        // Center
        {
            circle._center.x = c1->_center.x + (c2->_center.x-c3->_center.x);
            circle._center.y = c1->_center.y + (c2->_center.y-c3->_center.y);
            circle._center.x = utils::clip(circle._center.x, -circle._radius, this->_image_size.width+circle._radius);
            circle._center.y = utils::clip(circle._center.y, -circle._radius, this->_image_size.height+circle._radius);
        }
        break;
    default:
        std::cout << "ERROR: Unknown crossover feature id '" << crossover_feature << "'!" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

const std::shared_ptr<Circle> DifferentialCrossover::_findClosestCircle (const Circle &circle,
                                                                         const std::shared_ptr<Chromozome> ch)
{
    std::shared_ptr<Circle> closest;
    double closest_distance = DBL_MAX;

    for (size_t i = 0; i < ch->size(); ++i)
    {
        auto c = std::dynamic_pointer_cast<Circle>(ch->operator[](i));
        double distance = (c->_center.x-circle._center.x)*(c->_center.x-circle._center.x) +
                (c->_center.y-circle._center.y)*(c->_center.y-circle._center.y);

        if (distance < closest_distance)
        {
            closest_distance = distance;
            closest = c;
        }
    }

    return closest;
}


}
