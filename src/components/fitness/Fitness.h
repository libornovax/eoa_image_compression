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

extern int gpu_ms_total;
extern int cpu_ms_total;
extern int evals;

namespace eic {


template<typename CH>
void computeFitness (const std::vector<std::shared_ptr<CH>> &chromozomes, bool write_channels=false)
{
#ifdef USE_GPU
#ifdef MEASURE_TIME
    auto start1 = std::chrono::high_resolution_clock::now();
#endif
    computeFitnessGPU(chromozomes, write_channels);
#ifdef MEASURE_TIME
    auto end1 = std::chrono::high_resolution_clock::now();
    gpu_ms_total += std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count();
    evals++;
    std::cout << "GPU fitness computation time: " << (double(gpu_ms_total)/evals) << " ms" << std::endl;
#endif
#endif

#ifdef MEASURE_TIME
    auto start2 = std::chrono::high_resolution_clock::now();
#endif
    computeFitnessCPU(chromozomes, write_channels);
#ifdef MEASURE_TIME
    auto end2 = std::chrono::high_resolution_clock::now();
    cpu_ms_total += std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count();
    std::cout << "CPU fitness computation time: " << (double(cpu_ms_total)/evals) << " ms" << std::endl;
#endif
}


template<typename CH>
void computeFitness (const std::shared_ptr<CH> &ch, bool write_channels=false)
{
#ifdef USE_GPU
    computeFitnessGPU(ch, write_channels);
#else
    computeFitnessCPU(ch, write_channels);
#endif
}


}

#endif // FITNESS_H

