#include "ClassicEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "entities/OnePointCrossover.h"


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
        new_population[0] = this->_best_chromozome;

        // Evolve each individual in the population
        for (int i = 1; i < this->_population.size(); ++i)
        {
            // Tournament selection
            // Select 2 individuals for crossover
            const auto i1  = this->_tournamentSelection();
            auto offspring = this->_tournamentSelection()->clone();

            // Crossover
            OnePointCrossover crossover(image_size, i1);
            offspring->accept(crossover); // offspring is the editted chromozome

            // Mutation
            offspring->accept(mutator);

            offspring->computeDifference(this->_target);
            new_population[i] = offspring;

            // Check if this individual is not the best one so far
            if (offspring->getDifference() < this->_best_chromozome->getDifference())
            {
                std::cout << "[" << e << "] New best difference: " << offspring->getDifference() << std::endl;
                this->_best_chromozome = offspring;

                // Save the current image
                if (e-last_save > 100)
                {
                    last_save = e;
                    cv::Mat image = this->_best_chromozome->asImage(image_size);
                    cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(e) + ".png", image);
                }
            }
        }

        // Generational replacement with elitism (elitism is already taken care of)
        this->_population = new_population;
    }

    return *this->_best_chromozome->clone();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

std::shared_ptr<Chromozome> ClassicEA::_tournamentSelection ()
{
    // Select n random individuals for the tournament and select the best one from them
    // We imitate selecting n individuals by shuffling the indices in the population and taking the first
    // n individuals

    // Vector 0, 1, 2, ...
    std::vector<int> idxs(this->_population.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::random_shuffle(idxs.begin(), idxs.end());

    std::shared_ptr<Chromozome> best = this->_population[idxs[0]];
    for (int i = 1; i < Config::getParams().classic_ea.tournament_size; ++i)
    {
        // The difference of these individuals is already computed
        if (this->_population[idxs[i]]->getDifference() < best->getDifference())
        {
            // This is a better individual
            best = this->_population[idxs[i]];
        }
    }

    return best;
}


}
