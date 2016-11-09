#include "HillClimber.h"

#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"
#include "components/utils.h"


namespace eic {


HillClimber::HillClimber (const std::shared_ptr<Target> &target)
    : _target(target)
{
}


std::shared_ptr<Chromozome> HillClimber::run()
{
    // Initialize with a random chromozome
    this->_best_chromozome = Chromozome::randomChromozome(this->_target);

    {
        cv::Mat image = this->_best_chromozome->asImage();
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }

    Mutator mut(this->_target->image_size);
    int last_save = 0;
    for (int i = 0; i < Config::getParams().hill_climber.num_iterations; ++i)
    {        
        // Generate n mutated chromozomes and select the best one from them - Steepest Ascent Hill Climb
        double min_fitness = this->_best_chromozome->getFitness();
        int best = -1;
        std::vector<std::shared_ptr<Chromozome>> cloned_chromozomes(Config::getParams().hill_climber.pool_size);
        for (int c = 0; c < cloned_chromozomes.size(); ++c)
        {
            cloned_chromozomes[c] = this->_best_chromozome->clone();
            cloned_chromozomes[c]->accept(mut);

            if (cloned_chromozomes[c]->getFitness() < min_fitness)
            {
                // This one is the best - save it
                min_fitness = cloned_chromozomes[c]->getFitness();
                best = c;
            }
        }

        if (best >= 0)
        {
            // Replace the best chromozome, this one is better
            std::cout << "[" << i << "] Lowest difference: " << cloned_chromozomes[best]->getFitness() << " (" << cloned_chromozomes[best]->size() << ")" << std::endl;
            this->_best_chromozome = cloned_chromozomes[best];

            // Save the current image
            if (i-last_save > 200)
            {
                last_save = i;
                cv::Mat image = this->_best_chromozome->asImage();
                cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(i) + ".png", image);
            }
        }
    }

    return this->_best_chromozome->clone();
}



}
