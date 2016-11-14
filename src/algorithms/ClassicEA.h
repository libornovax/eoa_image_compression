//
// Libor Novak
// 11/07/2016
//

#ifndef CLASSICEA_H
#define CLASSICEA_H

#include <iostream>
#include "entities/NewChromozomePool.h"
#include "entities/Stats.h"


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
    virtual std::shared_ptr<Chromozome> run ();


protected:

    /**
     * @brief Initializes completely random population
     */
    virtual void _initializePopulation () final;

    /**
     * @brief Initializes new population with the elitist chromozome from the previous one
     * @param new_population Population to be editted
     */
    virtual void _initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const;

    /**
     * @brief Checks the new_population if there is a better individual then the currently best one (runs saving as well)
     * @param new_population
     * @param e Current epoch
     */
    virtual void _updateBestChromozome (const std::vector<std::shared_ptr<Chromozome>> &new_population, int e) final;

    /**
     * @brief Replaces the worst chromozome
     * @param new_population
     * @param e
     */
    virtual void _updateWorstChromozome (const std::vector<std::shared_ptr<Chromozome>> &new_population, int e) final;

    /**
     * @brief Replaces every n-th individual from the given population with a random new one
     * @param new_population
     */
    virtual void _refreshPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) final;

    /**
     * @brief Performs tournament selection of size given by the config
     * @param exclude_idx Index of individual to be excluded from the tournament
     * @return Index of an individual
     */
    virtual int _tournamentSelection (int exclude_idx=-1) const final;

    /**
     * @brief Performs crossover of the two offsprings - exchanges circles, which are at the same location
     * @param offspring1
     * @param offspring2
     */
    virtual void _onePointCrossover (std::shared_ptr<Chromozome> &offspring1, std::shared_ptr<Chromozome> &offspring2) final;

    /**
     * @brief Saves current population as images on grid
     * @param epoch Epoch number (for the filaname)
     */
    virtual void _saveCurrentPopulation (int epoch) final;

    /**
     * @brief Sorts the population by fitness in the ascending manner
     * @param population
     */
    static void _sortPopulation (std::vector<std::shared_ptr<Chromozome>> &population);

    /**
     * @brief Computes mean fitness of the given population
     * @param population
     * @return Mean fitness
     */
    static double _meanFitness (const std::vector<std::shared_ptr<Chromozome>> &population);

    /**
     * @brief Computes standard deviation of fitness in the population
     * @param population
     * @return
     */
    static double _stddevFitness (const std::vector<std::shared_ptr<Chromozome>> &population);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Target image channels, which we want to represent
    const std::shared_ptr<const Target> _target;
    // Population of candidate solutions
    std::vector<std::shared_ptr<Chromozome>> _population;
    // Best chromozome that we found so far
    std::shared_ptr<Chromozome> _best_chromozome;
    // Worst chromozome in the current population
    std::shared_ptr<Chromozome> _worst_chromozome;
    // Epoch, when we last saved the best chromozome
    int _last_save;
    // Asynchronous generator of new chromozomes for reinitialization
    NewChromozomePool _new_chromozome_pool;
    // Statistics of the evolution
    Stats _stats;

};


}


#endif // CLASSICEA_H
