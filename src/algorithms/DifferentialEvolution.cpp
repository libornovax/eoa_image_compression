#include "DifferentialEvolution.h"

#include <algorithm>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "shapes/Circle.h"
#include "components/Config.h"
#include "components/utils.h"
#include "entities/DifferentialCrossover.h"


namespace eic {


DifferentialEvolution::DifferentialEvolution (const std::vector<cv::Mat> &target)
    : _target(target)
{
    assert(target.size() == 3);
    assert(target[0].size() == target[1].size() && target[0].size() == target[2].size());
}


Chromozome DifferentialEvolution::run ()
{
    cv::Size image_size = this->_target[0].size();

    // Population must be greater than 3 otherwise the differential evolution won't work
    assert(Config::getParams().differential_evolution.population_size > 3);


    // Initialize the population
    for (int i = 0; i < Config::getParams().differential_evolution.population_size; ++i)
    {
        this->_population.push_back(Chromozome::randomChromozome(image_size));
    }

    this->_best_chromozome = this->_population[0];
    {
        cv::Mat image = this->_best_chromozome->asImage(image_size);
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }


    // Run the evolution
    Mutator mutator(image_size);
    int last_save = 0;
    for (int e = 0; e < Config::getParams().differential_evolution.num_epochs; ++e)
    {
        // Evolve each individual in the population
        for (int i = 0; i < this->_population.size(); ++i)
        {
            std::shared_ptr<Chromozome> clone = this->_population[i]->clone();

            // Selection
            std::vector<int> idxs = this->_selection(i);
            auto x1 = this->_population[idxs[0]];
            auto x2 = this->_population[idxs[1]];
            auto x3 = this->_population[idxs[2]];

            // Crossover
            DifferentialCrossover dc(image_size, x1, x2, x3);
            clone->accept(dc);

            // Mutation
            clone->accept(mutator);

            // Replacement
            if (clone->computeDifference(this->_target) < this->_population[i]->getDifference())
            {
                // The new chromozome is better -> replace the old one
                std::cout << "[" << e << "] (" << i << ") Replacing with difference: " << clone->getDifference() << std::endl;
                this->_population[i] = clone;


                // Update the current absolutely best chromozome
                if (clone->getDifference() < this->_best_chromozome->getDifference())
                {
                    this->_best_chromozome = clone;

                    // Save the current image
                    if (e-last_save > 200)
                    {
                        last_save = e;
                        cv::Mat image = this->_best_chromozome->asImage(image_size);
                        cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(e) + ".png", image);
                    }
                }
            }
        }
    }

    return *this->_best_chromozome->clone();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

std::vector<int> DifferentialEvolution::_selection (int i)
{
    // Vector 0, 1, 2, ...
    std::vector<int> idxs(this->_population.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    // Remove the current chromozome index
    idxs.erase(idxs.begin()+i);

    std::random_shuffle(idxs.begin(), idxs.end());

    // Return first 3 elements from the vector
    assert(idxs.size() >= 3);
    return { idxs[0], idxs[1], idxs[2] };
}


}
