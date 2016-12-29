#include "SteadyStateEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "shapes/Circle.h"
#include "components/fitness/Fitness.h"


namespace eic {


SteadyStateEA::SteadyStateEA(const std::shared_ptr<const Target> &target)
    : ClassicEA(target)
{

}


std::shared_ptr<Chromozome> SteadyStateEA::run ()
{
    this->_initializePopulation();

    // For the first half of the epochs we want to be evolving regions of the image, afer a half we
    // deactivate the rois
    for (auto ch: this->_population) ch->activateROI();

    // ROI activation makes chromozomes dirty - we need to recompute fitness
    computeFitness(this->_population, true);


    // Run the evolution
    for (int e = 0; e < Config::getParams().ea.num_epochs; ++e)
    {
        this->_stats.add(e, this->_best_chromozome->getFitness(), this->_worst_chromozome->getFitness(),
                         ClassicEA::_meanFitness(this->_population), ClassicEA::_stddevFitness(this->_population));

        if (e % POPULATION_SAVE_FREQUENCY == 0)
        {
            this->_saveCurrentPopulation(e);
            this->_stats.save();
        }

        // In the middle point of the evolution we deactivate the roi -> the algorithm will start focusing on
        // the whole image instead of the rois
        if (e == Config::getParams().ea.num_epochs/2)
        {
            std::cout << "DEACTIVATING REGIONS OF INTEREST OF CHROMOZOMES" << std::endl;
            for (auto ch: this->_population) ch->deactivateROI();
            computeFitness(this->_population, true);
        }


        // -- EVOLUTION -- //
        this->_steadyStateEpoch(e);


        // Sort the population by fitness
        ClassicEA::_sortPopulation(this->_population);

        this->_updateBestChromozome(e);
        this->_updateWorstChromozome(e);
    }

    return this->_best_chromozome->clone();
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void SteadyStateEA::_steadyStateEpoch (int epoch)
{
    Mutator mutator(this->_target->image_size);

    // Temporary new population
    std::vector<std::shared_ptr<Chromozome>> chromozomes;
    // Indices of the chromozomes that should be replaced with the new ones
    std::vector<int> indices;

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
        if (utils::makeMutation(Config::getParams().ea.crossover_prob))
        {
            ClassicEA::_onePointCrossover(offspring1, offspring2);
        }

        // Mutation
        offspring1->accept(mutator);
        offspring2->accept(mutator);

        // Put into the chromozome list for fitness computation (rendering)
        chromozomes.push_back(offspring1); indices.push_back(i1);
        chromozomes.push_back(offspring2); indices.push_back(i2);
    }


    // Compute fitness of all chromozomes
    if ((epoch+1) % POPULATION_SAVE_FREQUENCY == 0) computeFitness(chromozomes, true);
    else computeFitness(chromozomes);


    // Fill the new population with the new chromozomes, if they are better than the old ones
    for (int i = 0; i < chromozomes.size(); ++i)
    {
        if (chromozomes[i]->getFitness() < this->_population[indices[i]]->getFitness())
        {
            this->_population[indices[i]] = chromozomes[i];
        }
    }
}


}

