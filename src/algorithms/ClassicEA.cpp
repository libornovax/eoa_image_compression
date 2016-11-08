#include "ClassicEA.h"

#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"


namespace eic {


ClassicEA::ClassicEA (const std::vector<cv::Mat> &target)
    : _target(target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());
}


Chromozome ClassicEA::run ()
{
    cv::Size image_size = this->_target[0].size();


    // Initialize the population
    for (int i = 0; i < Config::getParams().classic_ea.population_size; ++i)
    {
        this->_population.push_back(Chromozome::randomChromozome(image_size));
        // Compute the score of each chromozome
        this->_population.back()->computeDifference(this->_target);
    }

    this->_best_chromozome = this->_population[0];
    {
        cv::Mat image = this->_best_chromozome->asImage(image_size);
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }


    // Run the evolution
    Mutator mutator(image_size);
    int last_save = 0;
    for (int e = 0; e < Config::getParams().classic_ea.num_epochs; ++e)
    {
        std::vector<std::shared_ptr<Chromozome>> new_population(this->_population.size());

        // Evolve each individual in the population
        for (int i = 0; i < this->_population.size(); ++i)
        {
            // Tournament selection

            // Crossover

            // Mutation



        }

        // Generational replacement with elitism
    }

    return *this->_best_chromozome->clone();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //




}
