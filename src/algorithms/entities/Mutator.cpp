#include "Mutator.h"

#include <iostream>
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
    chromozome.setDirty();

    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        if (utils::makeMutation(Config::getParams().mutator.shape_mutation_prob))
        {
            // Invoke mutation of the shape
            chromozome[i]->accept(*this);
        }
    }

#ifndef RENDER_AVERAGE
    // If we are rendering an ordered chromozome we also want to be able to change the order of the shapes
    // in the chromozome
    std::uniform_int_distribution<int> distch(0, chromozome.size()-1);

    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        if (utils::makeMutation(Config::getParams().mutator.shape_reorder_prob))
        {
            // Switch this shape with one on a different randomly selected position
            int replacee = distch(RGen::mt());
            auto tmp = chromozome[i];
            chromozome[i] = chromozome[replacee];
            chromozome[replacee] = tmp;
        }
    }
#endif
}


void Mutator::visit (Circle &circle)
{
    // Select one of the features of the circle, which will be mutated (we only want to mutate one per visit)
    std::uniform_int_distribution<int> distf(0, 5);  // RGBArc
    int mutated_feature = distf(RGen::mt());

    switch (mutated_feature)
    {
    case 4:
        // Radius
        if (utils::makeMutation(Config::getParams().mutator.radius_mutation_prob))
        {
            // Mutate the radius
            switch (circle.getSizeGroup())
            {
            case SizeGroup::SMALL:
                {
                    std::normal_distribution<double> dist(0, 1);
                    circle._radius += dist(RGen::mt());
                }
                break;
            case SizeGroup::MEDIUM:
                {
                    std::normal_distribution<double> dist(0, Config::getParams().mutator.radius_mutation_sdtddev/2);
                    circle._radius += dist(RGen::mt());
                }
                break;
            case SizeGroup::LARGE:
                {
                    std::normal_distribution<double> dist(0, Config::getParams().mutator.radius_mutation_prob);
                    circle._radius += dist(RGen::mt());
                }
                break;
            default:
                std::cout << "ERROR: Unknown SizeGroup" << std::endl;
                exit(EXIT_FAILURE);
                break;
            }
            auto minmax = Circle::radiusBounds(this->_image_size, circle.getSizeGroup());
            circle._radius = utils::clip(circle._radius, minmax.first, minmax.second);
        }
        break;
    case 5:
        // Center
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
            std::normal_distribution<double> dist(0, Config::getParams().mutator.position_mutation_stddev);
            circle._center.x += dist(RGen::mt());
            circle._center.y += dist(RGen::mt());
            circle._center.x = utils::clip(circle._center.x, -circle._radius, this->_image_size.width+circle._radius);
            circle._center.y = utils::clip(circle._center.y, -circle._radius, this->_image_size.height+circle._radius);
        }
        break;
    default:
        this->_mutateIShape(circle, mutated_feature);
        break;
    }
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Mutator::_mutateIShape (IShape &shape, int mutated_feature) const
{
    std::normal_distribution<double> distc(0, Config::getParams().mutator.color_mutation_stddev);

    switch (mutated_feature)
    {
    case 0:
        // Channel R
        if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
        {
            shape._r += distc(RGen::mt());
            shape._r = utils::clip(shape._r, 0, 255);
        }
        break;
    case 1:
        // Channel G
        if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
        {
            shape._g += distc(RGen::mt());
            shape._g = utils::clip(shape._g, 0, 255);
        }
        break;
    case 2:
        // Channel B
        if (utils::makeMutation(Config::getParams().mutator.color_mutation_prob))
        {
            shape._b += distc(RGen::mt());
            shape._b = utils::clip(shape._b, 0, 255);
        }
        break;
    case 3:
        // Channel alpha
        if (utils::makeMutation(Config::getParams().mutator.alpha_mutation_prob))
        {
            // Mutate the value of the alpha channel
            std::normal_distribution<double> dista(0, Config::getParams().mutator.alpha_mutation_stddev);
            shape._a += dista(RGen::mt());
#ifdef RENDER_AVERAGE
            // If rendering average, allow alpha to go higher - more pronounced color
            shape._a = utils::clip(shape._a, 30, 600);
#else
            shape._a = utils::clip(shape._a, 30, 60);
#endif
        }
        break;
    default:
        std::cout << "ERROR: Unknown mutated feature id '" << mutated_feature << "'!" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
}


}
