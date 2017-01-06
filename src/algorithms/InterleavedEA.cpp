#include "InterleavedEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "shapes/Circle.h"
#include "components/fitness/Fitness.h"


namespace eic {


InterleavedEA::InterleavedEA(const std::shared_ptr<const Target> &target)
    : SteadyStateEA(target),
      _hcp(Config::getParams().ea.population_size)
{

}


std::shared_ptr<Chromozome> InterleavedEA::run ()
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
        this->_stats.add(e, this->_best_chromozome->getBasicFitness(),
                         this->_worst_chromozome->getBasicFitness());

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
        if (e % Config::getParams().ea.interleaved_ea.hillclimb_frequency == 0)
        {
            // This is a hill climbing epoch
            std::cout << "[" << e << "] Hill climber epoch" << std::endl;
            this->_hillClimberEpoch(e);
        }
        else
        {
            // This is a steady state evolution epoch
            this->_steadyStateEpoch(e);
        }


        // Sort the population by fitness
        ClassicEA::_sortPopulation(this->_population);

        this->_updateBestChromozome(e);
        this->_updateWorstChromozome(e);
    }

    return this->_best_chromozome->clone();
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void InterleavedEA::_hillClimberEpoch (int epoch)
{
    // Run the whole population through hill climbing algorithm. The HillClimberPool modifies the population
    // in place - it does not create a new one!
    for (auto &ch: this->_population)
    {
        this->_hcp.addChromozome(ch);
    }

    this->_hcp.waitToFinish();

    // Compute fitness of all chromozomes
    if ((epoch+1) % POPULATION_SAVE_FREQUENCY == 0) computeFitness(this->_population, true);
    else computeFitness(this->_population);
}



}
