//
// Libor Novak
// 12/29/2016
//

#ifndef FITNESS_H
#define FITNESS_H

#include "cpu/CPUFitness.h"
#ifdef USE_GPU
#include "gpu/GPUFitness.h"
#endif


namespace eic {


template<typename CH>
void computeFitness (const std::vector<std::shared_ptr<CH>> &chromozomes, bool write_channels=false)
{
#ifdef USE_GPU
    computeFitnessCPU(chromozomes, write_channels);
    computeFitnessGPU(chromozomes, write_channels);
#else
    computeFitnessCPU(chromozomes, write_channels);
#endif
}


template<typename CH>
void computeFitness (const std::shared_ptr<CH> &ch, bool write_channels=false)
{
#ifdef USE_GPU
    computeFitnessCPU(ch, write_channels);
    computeFitnessGPU(ch, write_channels);
#else
    computeFitnessCPU(ch, write_channels);
#endif
}


}

#endif // FITNESS_H

