#include "HillClimber.h"

#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"
#include "components/utils.h"
#include "components/fitness/Fitness.h"


namespace eic {


HillClimber::HillClimber (bool save_and_print)
    : _save_and_print(save_and_print)
{
}


std::shared_ptr<Chromozome> HillClimber::run (const std::shared_ptr<Chromozome> &chromozome)
{
    assert(chromozome != nullptr);

    // Initialize the best chromozome with the current one
    this->_best_chromozome = chromozome; computeFitness(this->_best_chromozome, true);

    if (this->_save_and_print)
    {
        cv::Mat image = this->_best_chromozome->asImage();
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }


    // -- RUN THE HILL CLIMBER -- //
    Mutator mut(chromozome->getTarget()->image_size);
    for (int i = 0; i < Config::getParams().hill_climber.num_iterations; ++i)
    {
        if (this->_save_and_print)
        {
            this->_stats.add(i, this->_best_chromozome->getFitness());
            if (i % 100 == 0) this->_stats.save();
        }

        // Generate n mutated chromozomes and select the best one from them - Steepest Ascent Hill Climb
        std::vector<std::shared_ptr<Chromozome>> chromozomes(Config::getParams().hill_climber.pool_size);
        for (int c = 0; c < Config::getParams().hill_climber.pool_size; ++c)
        {
            chromozomes[c] = this->_best_chromozome->clone();
            chromozomes[c]->accept(mut);
        }

        // Compute fitness of all chromozomes
        computeFitness(chromozomes);

        for (int c = 0; c < Config::getParams().hill_climber.pool_size; ++c)
        {
            if (chromozomes[c]->getFitness() < this->_best_chromozome->getFitness())
            {
                // This one is better - save it
                this->_best_chromozome = chromozomes[c];

                if (this->_save_and_print)
                {
                    std::cout << "[" << i << "] Lowest difference: " << this->_best_chromozome->getFitness() << " (" << this->_best_chromozome->size() << ")" << std::endl;
                }
            }
        }

        if (this->_save_and_print && i % 200 == 0)
        {
            // Save the current best image
            computeFitness(this->_best_chromozome, true);  // Render the image channels
            cv::Mat image = this->_best_chromozome->asImage();
            cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(i) + ".png", image);
        }
    }

    return this->_best_chromozome->clone();
}



}
