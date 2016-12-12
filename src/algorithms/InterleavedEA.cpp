#include "InterleavedEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "shapes/Circle.h"


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

    // Run the evolution
    for (int e = 0; e < Config::getParams().ea.num_epochs; ++e)
    {
        this->_stats.add(e, this->_best_chromozome->getFitness(), this->_worst_chromozome->getFitness(),
                         ClassicEA::_meanFitness(this->_population),ClassicEA::_stddevFitness(this->_population));

        if ((e < 1000 && e % 50 == 0) || (e >= 1000 && e % 100 == 0))
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
        }


        // -- EVOLUTION -- //
        if (e % Config::getParams().ea.interleaved_ea.hillclimb_frequency == 0)
        {
            // This is a hill climbing epoch
            std::cout << "[" << e << "] Hill climber epoch" << std::endl;
            this->_hillClimberEpoch();
        }
        else
        {
            // This is a steady state evolution epoch
            this->_steadyStateEpoch();
        }


        // Sort the population by fitness
        ClassicEA::_sortPopulation(this->_population);

        this->_updateBestChromozome(e);
        this->_updateWorstChromozome(e);
    }

    return this->_best_chromozome->clone();
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void InterleavedEA::_hillClimberEpoch ()
{
    // Run the whole population through hill climbing algorithm
    for (auto &ch: this->_population)
    {
        this->_hcp.addChromozome(ch);
    }

    this->_hcp.waitToFinish();
}



}
