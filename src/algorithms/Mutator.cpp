#include "Mutator.h"

#include "components/utils.h"
#include "components/RGen.h"
#include "shapes/Circle.h"
#include "components/Chromozome.h"
#include "components/Config.h"


namespace eic {


Mutator::Mutator (const cv::Size &image_size)
    : _image_size(image_size)
{

}


void Mutator::visit (Chromozome &chromozome)
{
    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        if (chromozome[i]->isNew())
        {
            // This is a new shape in the chromozome, it should mutate every time to be settled faster
            chromozome[i]->accept(*this);
        }
        else if (utils::makeMutation(Config::getParams().mutator.shape_mutation_prob))
        {
            // Invoke mutation of the shape
            chromozome[i]->accept(*this);
        }
    }
}


void Mutator::visit (Circle &circle)
{
    if (utils::makeMutation(Config::getParams().mutator.radius_mutation_prob))
    {
        // Mutate the radius
        std::normal_distribution<double> dist (0, Config::getParams().mutator.radius_mutation_sdtddev);
        circle._radius += dist(RGen::mt());
        circle._radius = utils::clip(circle._radius, 2, 10000); // Must be positive
    }

    if (utils::makeMutation(Config::getParams().mutator.position_reinitialization_prob))
    {
        // Generate a completely new position for the center
        std::uniform_int_distribution<int> distcx(0, this->_image_size.width);
        std::uniform_int_distribution<int> distcy(0, this->_image_size.height);
        circle._center.x = distcx(RGen::mt());
        circle._center.y = distcy(RGen::mt());
    }
    else if (utils::makeMutation(Config::getParams().mutator.position_mutation_prob))
    {
        // Mutate the position of the center
        std::normal_distribution<double> dist (0, Config::getParams().mutator.position_mutation_stddev);
        circle._center.x += dist(RGen::mt());
        circle._center.y += dist(RGen::mt());
        circle._center.x = utils::clip(circle._center.x, -circle._radius, this->_image_size.width+circle._radius);
        circle._center.y = utils::clip(circle._center.y, -circle._radius, this->_image_size.height+circle._radius);
    }

    this->_mutateIShape(circle);
}


void Mutator::_mutateIShape (IShape &shape) const
{
    std::normal_distribution<double> distc(0, Config::getParams().mutator.color_mutation_stddev);
    if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
    {
        shape._r += distc(RGen::mt());
        shape._r = utils::clip(shape._r, 0, 255);
    }
    if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
    {
        shape._g += distc(RGen::mt());
        shape._g = utils::clip(shape._g, 0, 255);
    }
    if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
    {
        shape._b += distc(RGen::mt());
        shape._b = utils::clip(shape._b, 0, 255);
    }

    if (utils::makeMutation(Config::getParams().mutator.alpha_mutation_prob))
    {
        // Mutate the value of the alpha channel
        std::normal_distribution<double> dista(0, Config::getParams().mutator.alpha_mutation_stddev);
        shape._a += dista(RGen::mt());
        shape._a = utils::clip(shape._a, 20, 80);
    }
}


}
