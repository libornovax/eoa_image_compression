#include "SteadyStateEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "shapes/Circle.h"


namespace eic {


SteadyStateEA::SteadyStateEA(const std::shared_ptr<const Target> &target)
    : ClassicEA(target)
{

}


std::shared_ptr<Chromozome> SteadyStateEA::run ()
{
    // Start chromozome generation
    this->_new_chromozome_pool.launch();

    this->_initializePopulation();

    // For the first half of the epochs we want to be evolving regions of the image, afer a half we
    // deactivate the rois
    for (auto ch: this->_population) ch->activateROI();

    // Run the evolution
    Mutator mutator(this->_target->image_size);
    for (int e = 0; e < Config::getParams().classic_ea.num_epochs; ++e)
    {
        this->_stats.add(e, this->_best_chromozome->getFitness(), this->_worst_chromozome->getFitness(),
                         ClassicEA::_meanFitness(this->_population), ClassicEA::_stddevFitness(this->_population));

        if ((e < 1000 && e % 50 == 0) || (e >= 1000 && e % 100 == 0))
        {
            this->_saveCurrentPopulation(e);
            this->_stats.save();
        }

        // In the middle point of the evolution we deactivate the roi -> the algorithm will start focusing on
        // the whole image instead of the rois
        if (e == Config::getParams().classic_ea.num_epochs/2)
        {
            for (auto ch: this->_population) ch->deactivateROI();
        }


        // -- EVOLUTION -- //
        std::vector<std::shared_ptr<Chromozome>> new_population;
        this->_initializeNewPopulation(new_population);

        // Population size/2 times perform tournament selection, crossover, mutation and replace the parents
        // by their children only if they are better
        for (int i = 0; i < this->_population.size()/2; ++i)
        {
            // Tournament selection
            // Select 2 individuals for crossover
            int i1 = this->_tournamentSelection();
            int i2 = this->_tournamentSelection(i1);

            // Careful! We have to clone here!!!
            auto offspring1 = this->_population[i1]->clone();
            auto offspring2 = this->_population[i2]->clone();

            // Crossover
            if (utils::makeMutation(Config::getParams().classic_ea.crossover_prob))
            {
                ClassicEA::_onePointCrossover(offspring1, offspring2);
            }

            // Mutation
            offspring1->accept(mutator);
            offspring2->accept(mutator);

            // Put the offsprings into the new population if they are better than their parents and better
            // than the solutions that can be already in the new population from previous crossovers
            if (offspring1->getFitness() < this->_population[i1]->getFitness() &&
                    (!new_population[i1] || offspring1->getFitness() < new_population[i1]->getFitness()))
            {
                new_population[i1] = offspring1;
            }
            if (offspring2->getFitness() < this->_population[i2]->getFitness() &&
                    (!new_population[i2] || offspring2->getFitness() < new_population[i2]->getFitness()))
            {
                new_population[i2] = offspring2;
            }
        }

        // Fill the gaps - the individuals, which were not replaced have to be put to the new population
        for (int i = 0; i < this->_population.size(); ++i)
        {
            if (!new_population[i])
            {
                new_population[i] = this->_population[i];
            }
        }

        // All chromozomes age
        for (auto ch: new_population) ch->birthday();

        // Sort the population by fitness
        ClassicEA::_sortPopulation(new_population);

        this->_updateBestChromozome(new_population, e);
        this->_updateWorstChromozome(new_population, e);

        // Replace some of the individuals with random new ones to keep diversity in the population
        if (e > 0 && e % Config::getParams().classic_ea.refresh_interval == 0)
        {
            std::cout << "AGES: "; for (auto ch: new_population) std::cout << ch->getAge() << " "; std::cout << std::endl;
            this->_refreshPopulation(new_population);
        }

        // Replace the population with the new steady state one
        this->_population = new_population;
    }

    // Shut down the chromozome generator
    this->_new_chromozome_pool.shutDown();

    return this->_best_chromozome->clone();
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void SteadyStateEA::_initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const
{
    new_population.resize(this->_population.size());
}



}

