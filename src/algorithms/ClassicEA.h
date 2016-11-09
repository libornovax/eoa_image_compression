
//
// Libor Novak
// 11/07/2016
//

#ifndef CLASSICEA_H
#define CLASSICEA_H

#include <iostream>
#include "components/Chromozome.h"


namespace eic {


/**
 * @brief The ClassicEA class
 * Classic evolutionary algorithm with tournament selection, crossover, mutation and generational replacement
 * with elitism
 */
class ClassicEA
{
public:

    ClassicEA (const std::shared_ptr<const Target> &target);


    /**
     * @brief Run the hill differential evolution process and return the best found chromozome
     */
    std::shared_ptr<Chromozome> run ();


private:

    /**
     * @brief Performs tournament selection of size given by the config
     * @param exclude_idx Index of individual to be excluded from the tournament
     * @return Index of an individual
     */
    int _tournamentSelection (int exclude_idx=-1);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Target image channels, which we want to represent
    const std::shared_ptr<const Target> _target;
    // Population of candidate solutions
    std::vector<std::shared_ptr<Chromozome>> _population;
    // Best chromozome that we found so far
    std::shared_ptr<Chromozome> _best_chromozome;

};


}


#endif // CLASSICEA_H
