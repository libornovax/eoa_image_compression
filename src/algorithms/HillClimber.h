//
// Libor Novak
// 10/12/2016
//

#ifndef HILLCLIMBER_H
#define HILLCLIMBER_H

#include <iostream>
#include "components/Chromozome.h"
#include "entities/Stats.h"


namespace eic {


/**
 * @brief The HillClimber class
 * An implementation of a hill climber algorithm for image representation optimization
 */
class HillClimber
{
public:

    HillClimber (bool save_and_print=true);


    /**
     * @brief Run the hill climber algorithm and return the best found chromozome
     * @param chromozome Chromozome to be optimized
     */
    std::shared_ptr<Chromozome> run (const std::shared_ptr<Chromozome> &chromozome);


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Turns on and off the prints and saving of the optimized chromozone
    bool _save_and_print;
    // Best chromozome that we found so far
    std::shared_ptr<Chromozome> _best_chromozome;
    // Statistics of the evolution
    Stats _stats;

};


}


#endif // HILLCLIMBER_H
