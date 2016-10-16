#include "HillClimber.h"

#include <opencv2/highgui/highgui.hpp>
#include "Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"
#include "components/utils.h"


namespace eic {


HillClimber::HillClimber (const std::vector<cv::Mat> &target)
    : _target(target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());
}


Chromozome HillClimber::run ()
{
    cv::Size image_size = this->_target[0].size();

    // Initialize with a random chromozome
    this->_best_chromozome = Chromozome::randomChromozome(image_size);

    Mutator mut(image_size);
    for (int i = 0; i < Config::getParams().hill_climber.num_iterations; ++i)
    {
        // Every x iterations add a new shape to the chromozome - we want to be gradually increasing the
        // complexity of the chromozome
        if (utils::makeMutation(Config::getParams().hill_climber.shape_add_prob))
        {
            // Save the current image
            {
                cv::Mat image = this->_best_chromozome.asImage(image_size);
                cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(i) + ".png", image);
            }

            this->_best_chromozome.addRandomShape(image_size);
            this->_best_chromozome.computeDifference(this->_target);
        }

        // Generate n mutated chromozomes and select the best one from them - Steepest Ascent Hill Climb
        double min_difference = this->_best_chromozome.getDifference();
        int best = -1;
        std::vector<Chromozome> cloned_chromozomes(Config::getParams().hill_climber.pool_size);
        for (int c = 0; c < cloned_chromozomes.size(); ++c)
        {
            cloned_chromozomes[c] = this->_best_chromozome.clone();
            cloned_chromozomes[c].accept(mut);

            if (cloned_chromozomes[c].computeDifference(this->_target) < min_difference)
            {
                // This one is the best - save it
                min_difference = cloned_chromozomes[c].getDifference();
                best = c;
            }
        }

        if (best >= 0)
        {
            // Replace the best chromozome, this one is better
            std::cout << "[" << i << "] Lowest difference: " << cloned_chromozomes[best].getDifference() << " (" << cloned_chromozomes[best].size() << ")" << std::endl;
            this->_best_chromozome = cloned_chromozomes[best];
        }
    }

    return this->_best_chromozome.clone();
}



}
