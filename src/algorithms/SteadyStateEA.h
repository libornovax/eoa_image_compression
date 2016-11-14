//
// Libor Novak
// 11/14/2016
//

#ifndef STEADYSTATEEA_H
#define STEADYSTATEEA_H

#include "ClassicEA.h"


namespace eic {


/**
 * @brief The SteadyStateEA class
 * Steady state evolutionary algorithm, which replaces parents only if the new generated chromozomes are
 * better than the ones in the population
 */
class SteadyStateEA : public ClassicEA
{
public:

    SteadyStateEA (const std::shared_ptr<const Target> &target);


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

};


}


#endif // STEADYSTATEEA_H
