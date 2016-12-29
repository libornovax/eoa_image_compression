//
// Libor Novak
// 12/29/2016
//

#ifndef FITNESS_H
#define FITNESS_H

#include "cpu/CPUFitness.h"


namespace eic {


template<typename CH>
void computeFitness (const std::vector<std::shared_ptr<CH>> &chromozomes, bool write_channels=false)
{
#ifdef USE_GPU
#else
    CPUFitness::computeFitness(chromozomes, write_channels);
#endif
}


template<typename CH>
void computeFitness (const std::shared_ptr<CH> &ch, bool write_channels=false)
{
#ifdef USE_GPU
#else
    CPUFitness::computeFitness(ch, write_channels);
#endif
}


}

#endif // FITNESS_H

