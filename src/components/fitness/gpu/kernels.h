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
void populationFitness (__uint8_t *g_target, unsigned int width, unsigned int height, float *g_population,
                        unsigned int population_size, unsigned int chromozome_length, float *g_out_fitness,
                        float *s_canvas);


}


#endif // KERNELS_H

