#include "HillClimber.h"

#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"
#include "components/utils.h"


namespace eic {


HillClimber::HillClimber (bool save_and_print)
    : _save_and_print(save_and_print)
{
}


std::shared_ptr<Chromozome> HillClimber::run (const std::shared_ptr<Chromozome> &chromozome)
{
    assert(chromozome != nullptr);

    // Initialize the best chromozome with the current one
    this->_best_chromozome = chromozome;

    if (this->_save_and_print)
    {
        cv::Mat image = this->_best_chromozome->asImage();
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }


    Mutator mut(chromozome->getTarget()->image_size);
    int last_save = 0;

    for (int i = 0; i < Config::getParams().hill_climber.num_iterations; ++i)
    {
        if (this->_save_and_print)
        {
            this->_stats.add(i, this->_best_chromozome->getFitness());
            if (i % 100 == 0) this->_stats.save();
        }

        // Generate n mutated chromozomes and select the best one from them - Steepest Ascent Hill Climb
        double min_fitness = this->_best_chromozome->getFitness();
        std::shared_ptr<Chromozome> best;
        for (int c = 0; c < Config::getParams().hill_climber.pool_size; ++c)
        {
            auto cloned_chromozome = this->_best_chromozome->clone();
            cloned_chromozome->accept(mut);

            if (cloned_chromozome->getFitness() < min_fitness)
            {
                // This one is better - save it
                min_fitness = cloned_chromozome->getFitness();
                best = cloned_chromozome;
            }
        }

        if (best != nullptr)
        {
            // Replace the best chromozome, this one is better
            this->_best_chromozome = best;

            if (this->_save_and_print)
            {
                std::cout << "[" << i << "] Lowest difference: " << best->getFitness() << " (" << best->size() << ")" << std::endl;

                // Save the current image
                if (i-last_save > 200)
                {
                    last_save = i;
                    cv::Mat image = this->_best_chromozome->asImage();
                    cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(i) + ".png", image);
                }
            }
        }
    }

    return this->_best_chromozome->clone();
}



}
