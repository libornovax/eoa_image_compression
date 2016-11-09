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
//        {
//            int i = 0;
//            for (auto ch: this->_population)
//            {
//                cv::Mat image = ch->asImage(image_size);
//                cv::imshow("individual " + std::to_string(i++), image);
//            }
//            cv::waitKey();
//        }

        std::vector<std::shared_ptr<Chromozome>> new_population(this->_population.size());
        new_population[0] = this->_best_chromozome;

        // Evolve each individual in the population
        for (int i = 1; i < this->_population.size(); ++i)
        {
            // Tournament selection
            // Select 2 individuals for crossover
//            int i1 = this->_tournamentSelection();
            int i2 = this->_tournamentSelection();
//            std::cout << "Crossover individuals: [" << i1 << "] [" << i2 << "]" << std::endl;

            auto offspring = this->_population[i]->clone();

            // Crossover
            if (utils::makeMutation(Config::getParams().classic_ea.crossover_prob))
            {
                OnePointCrossover crossover(image_size, this->_population[i2]);
                offspring->accept(crossover); // offspring is the editted chromozome
            }

            // Mutation
            offspring->accept(mutator);

            offspring->computeDifference(this->_target);

            // Replace the current individual in the population only if the offspring is better
            if (offspring->getDifference() < this->_population[i]->getDifference())
            {
                new_population[i] = offspring;
            }
            else
            {
                new_population[i] = this->_population[i];
            }

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

int ClassicEA::_tournamentSelection(int exclude_idx)
{
    // Select n random individuals for the tournament and select the best one from them
    // We imitate selecting n individuals by shuffling the indices in the population and taking the first
    // n individuals

    // Vector 0, 1, 2, ...
    std::vector<int> idxs(this->_population.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    // Erase the index we want to exclude
    if (exclude_idx >= 0 && exclude_idx < idxs.size())
    {
        idxs.erase(idxs.begin()+exclude_idx);
    }

    std::random_shuffle(idxs.begin(), idxs.end());

    // Take the first tournament_size indices
    std::vector<std::pair<int, double>> selected;
    for (int i = 0; i < Config::getParams().classic_ea.tournament_size; ++i)
    {
        selected.emplace_back(idxs[i], this->_population[idxs[i]]->getDifference());
    }

    // Order them by ascending difference
    std::sort(selected.begin(), selected.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b){ return a.second < b.second; });

    for (auto sel: selected)
    {
        if (utils::makeMutation(0.5))
        {
            return sel.first;
        }
    }

    return selected.back().first;
}


}
