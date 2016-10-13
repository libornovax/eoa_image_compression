#include "HillClimber.h"

#include "Mutator.h"
#include "shapes/Circle.h"


namespace eic {


HillClimber::HillClimber (const std::vector<cv::Mat> &target)
    : _target(target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());
}


Chromozome HillClimber::run ()
{
    // TODO!! This must be done differently!
    for (int i = 0; i < 100; ++i)
    {
        this->_best_chromozome.chromozome().push_back(eic::Circle::randomCircle(this->_target[0].size()));
    }

    Mutator mut;

    for (int i = 0; i < 1000; ++i)
    {
        Chromozome cloned = this->_best_chromozome.clone();
        // Mutate the chromozome
        cloned.accept(mut);

        if (cloned.computeDifference(this->_target) < this->_best_chromozome.getDifference())
        {
            // Replace the best chromozome, this one is better
            std::cout << "[" << i << "] Lowest difference: " << cloned.getDifference() << std::endl;
            this->_best_chromozome = cloned;
        }
    }

    return this->_best_chromozome.clone();
}



}
