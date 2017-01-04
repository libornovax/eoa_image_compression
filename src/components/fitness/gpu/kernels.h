//
// Libor Novak
// 01/03/2017
//

#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace eic {


__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int offset, int population_size, int chromozome_length,
                        float *g_out_fitness, int *g_all_canvas);

__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int offset, int population_size, int chromozome_length,
                        float *g_out_fitness);

}


#endif // KERNELS_H

