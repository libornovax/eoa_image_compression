#include "Mutator.h"

#include "components/utils.h"
#include "components/RGen.h"
#include "shapes/Circle.h"
#include "components/Chromozome.h"
#include "components/Config.h"


namespace eic {


Mutator::Mutator ()
{

}


void Mutator::visit (Chromozome &chromozome)
{
    for (size_t i = 0; i < chromozome.size(); ++i)
    {
        // Invoke mutation of each shape
        if (utils::makeMutation(Config::getParams().mutator.shape_mutation_prob))
        {
            chromozome[i]->accept(*this);
        }
    }
}


void Mutator::visit (Circle &circle)
{
    // Probability of mutation
    std::uniform_real_distribution<double> distp(0.0, 1.0);

    std::normal_distribution<double> distr (0, 5); // mean, stddev
    if (distp(RGen::mt()) < 0.2) circle._radius += distr(RGen::mt());
    circle._radius = utils::clip(circle._radius, 0, 10000); // Must be positive

    std::normal_distribution<double> distc (0, 10); // mean, stddev
    if (distp(RGen::mt()) < 0.2)
    {
        circle._center.x += distc(RGen::mt());
        circle._center.y += distc(RGen::mt());
    }

    this->_mutateIShape(circle);
}


void Mutator::_mutateIShape (IShape &shape) const
{
    // Probability of mutation
    std::uniform_real_distribution<double> distp(0.0, 1.0);

    std::normal_distribution<double> dist(0, 10); // mean, stddev
    if (distp(RGen::mt()) < 0.2) shape._r += dist(RGen::mt());
    if (distp(RGen::mt()) < 0.2) shape._g += dist(RGen::mt());
    if (distp(RGen::mt()) < 0.2) shape._b += dist(RGen::mt());
    if (distp(RGen::mt()) < 0.2) shape._a += dist(RGen::mt());

    // Correct the values to be in the interval [0,255]
    shape._r = utils::clip(shape._r, 0, 255);
    shape._g = utils::clip(shape._g, 0, 255);
    shape._b = utils::clip(shape._b, 0, 255);
    // Correct the value to be in the interval [0,100]
    shape._a = utils::clip(shape._a, 0, 100);
}


}
