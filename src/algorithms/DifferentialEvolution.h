//
// Libor Novak
// 11/07/2016
//

#ifndef DIFFERENTIALEVOLUTION_H
#define DIFFERENTIALEVOLUTION_H


#include <iostream>
#include "components/Chromozome.h"


namespace eic {


/**
 * @brief The DifferentialEvolution class
 * Differential evolution algorithm for image compression
 */
class DifferentialEvolution
{
public:

    DifferentialEvolution (const std::vector<cv::Mat> &target);


    /**
     * @brief Run the hill differential evolution process and return the best found chromozome
     */
    Chromozome run ();


private:

    /**
     * @brief Selects 3 random candidates for differential crossover, which are different from i
     * @param i Index of the chromozome being mutated
     * @return vector of length 3 with indices of random individuals to use
     */
    std::vector<int> _selection (int i);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Target image channels, which we want to represent
    const std::vector<cv::Mat> _target;
    // Population of candidate solutions
    std::vector<std::shared_ptr<Chromozome>> _population;
    // Best chromozome that we found so far
    std::shared_ptr<Chromozome> _best_chromozome;

};


}

#endif // DIFFERENTIALEVOLUTION_H
