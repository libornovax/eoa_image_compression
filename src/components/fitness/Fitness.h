//
// Libor Novak
// 12/29/2016
//

#ifndef FITNESS_H
#define FITNESS_H

#include <chrono>
#include "cpu/CPUFitness.h"
#ifdef USE_GPU
#include "gpu/GPUFitness.h"
#endif


namespace eic {


template<typename CH>
void computeFitness (const std::vector<std::shared_ptr<CH>> &chromozomes, bool write_channels=false)
{
#ifdef MEASURE_TIME
    auto start = std::chrono::high_resolution_clock::now();
#endif

#ifdef USE_GPU
//    computeFitnessCPU(chromozomes, write_channels);
    computeFitnessGPU(chromozomes, write_channels);
#else
    computeFitnessCPU(chromozomes, write_channels);
#endif

#ifdef MEASURE_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Fitness computation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;
#endif
}


template<typename CH>
void computeFitness (const std::shared_ptr<CH> &ch, bool write_channels=false)
{
#ifdef MEASURE_TIME
    auto start = std::chrono::high_resolution_clock::now();
#endif

#ifdef USE_GPU
//    computeFitnessCPU(ch, write_channels);
    computeFitnessGPU(ch, write_channels);
#else
    computeFitnessCPU(ch, write_channels);
#endif

#ifdef MEASURE_TIME
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Fitness computation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << " ms" << std::endl;
#endif
}


}

#endif // FITNESS_H

