
//
// Libor Novak
// 11/07/2016
//

#ifndef CLASSICEA_H
#define CLASSICEA_H

#include <iostream>
#include "entities/NewChromozomePool.h"


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
     * @brief Initializes completely random population
     */
    void _initializePopulation ();

    /**
     * @brief Initializes new population with the elitist chromozome from the previous one
     * @param new_population Population to be editted
     */
    void _initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const;

    /**
     * @brief Checks the new_population if there is a better individual then the currently best one (runs saving as well)
     * @param new_population
     * @param e Current epoch
     */
    void _updateBestChromozome (const std::vector<std::shared_ptr<Chromozome>> &new_population, int e);

    /**
     * @brief Replaces every n-th individual from the given population with a random new one
     * @param new_population
     */
    void _refreshPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population);

    /**
     * @brief Performs tournament selection of size given by the config
     * @param exclude_idx Index of individual to be excluded from the tournament
     * @return Index of an individual
     */
    int _tournamentSelection (int exclude_idx=-1) const;

    /**
     * @brief Performs crossover of the two offsprings - exchanges circles, which are at the same location
     * @param offspring1
     * @param offspring2
     */
    void _onePointCrossover (std::shared_ptr<Chromozome> &offspring1, std::shared_ptr<Chromozome> &offspring2);

    /**
     * @brief Saves current population as images on grid
     * @param epoch Epoch number (for the filaname)
     */
    void _saveCurrentPopulation (int epoch);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Target image channels, which we want to represent
    const std::shared_ptr<const Target> _target;
    // Population of candidate solutions
    std::vector<std::shared_ptr<Chromozome>> _population;
    // Best chromozome that we found so far
    std::shared_ptr<Chromozome> _best_chromozome;
    // Epoch, when we last saved the best chromozome
    int _last_save;
    // Asynchronous generator of new chromozomes for reinitialization
    NewChromozomePool _new_chromozome_pool;

};


}


#endif // CLASSICEA_H
