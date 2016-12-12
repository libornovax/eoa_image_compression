//
// Libor Novak
// 12/12/2016
//

#ifndef INTERLEAVEDEA_H
#define INTERLEAVEDEA_H

#include "ClassicEA.h"
#include "entities/HillClimberPool.h"


namespace eic {


/**
 * @brief The InterleavedEA class
 * The Interleaved Evolutionary algorithm - we mix hill climber with steady state evolution. A random
 * population is initialized, then it is optimized by hill climbing and after a number of hill climbing steps
 * a crossover step is applied. These two steps keep interleaving for the whole time of the evolution.
 */
class InterleavedEA : public ClassicEA
{
public:

    InterleavedEA (const std::shared_ptr<const Target> &target);


    /**
     * @brief Run the hill differential evolution process and return the best found chromozome
     */
    virtual std::shared_ptr<Chromozome> run () override final;


protected:

    /**
     * @brief Initializes new population with NULL pointers
     * @param new_population Population to be editted
     */
    virtual void _initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const override final;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Thread pool that runs hill climber optimization on the given chromozomes
    HillClimberPool _hcp;

};


}

#endif // INTERLEAVEDEA_H
