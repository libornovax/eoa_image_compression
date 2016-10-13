//
// Libor Novak
// 10/12/2016
//

#ifndef HILLCLIMBER_H
#define HILLCLIMBER_H

#include <iostream>
#include "components/Chromozome.h"


namespace eic {


/**
 * @brief The HillClimber class
 * An implementation of a hill climber algorithm for image representation optimization
 */
class HillClimber
{
public:

    HillClimber (const std::vector<cv::Mat> &target);


    /**
     * @brief Run the hill climber algorithm and return the best found chromozome
     */
    Chromozome run ();


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Target image channels, which we want to represent
    const std::vector<cv::Mat> _target;
    // Best chromozome that we found so far
    Chromozome _best_chromozome;

};


}


#endif // HILLCLIMBER_H
