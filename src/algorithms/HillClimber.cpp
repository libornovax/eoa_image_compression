#include "HillClimber.h"

#include "Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"


namespace eic {


HillClimber::HillClimber (const std::vector<cv::Mat> &target)
    : _target(target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());
}


Chromozome HillClimber::run ()
{
    // Initialize with a random chromozome
    this->_best_chromozome = Chromozome::randomChromozome(this->_target[0].size());

    Mutator mut;
    for (int i = 0; i < Config::getParams().hill_climber.num_iterations; ++i)
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
