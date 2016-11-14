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

    // Run the evolution
    Mutator mutator(this->_target->image_size);
    for (int e = 0; e < Config::getParams().classic_ea.num_epochs; ++e)
    {
        if (e % 1 == 0)
        {
            this->_saveCurrentPopulation(e);
        }

        std::vector<std::shared_ptr<Chromozome>> new_population;
        this->_initializeNewPopulation(new_population);

        // -- EVOLUTION -- //
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

            // Put the offsprings into the new population if they are better than their parents
            if (offspring1->getFitness() < this->_population[i1]->getFitness())
            {
                new_population[i1] = offspring1;
            }
            if (offspring2->getFitness() < this->_population[i2]->getFitness())
            {
                new_population[i2] = offspring2;
            }
        }

        // All chromozomes age
        for (auto ch: new_population) ch->birthday();

        // Sort the population by fitness
        std::sort(new_population.begin(), new_population.end(),
                  [] (const std::shared_ptr<Chromozome> &ch1, const std::shared_ptr<Chromozome> &ch2) {
            return ch1->getFitness() < ch2->getFitness();
        });


        this->_updateBestChromozome(new_population, e);

        // Replace some of the individuals with random new ones to keep diversity in the population
        if (e > 0 && e % Config::getParams().classic_ea.refresh_interval == 0)
        {
            std::cout << "AGES: "; for (auto ch: new_population) std::cout << ch->getAge() << " "; std::cout << std::endl;
            this->_refreshPopulation(new_population);
        }

        // Generational replacement with elitism (elitism is already taken care of)
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

    // Clone the whole population
    for (int i = 0; i < this->_population.size(); ++i)
    {
        new_population[i] = this->_population[i]->clone();
    }
}



}

