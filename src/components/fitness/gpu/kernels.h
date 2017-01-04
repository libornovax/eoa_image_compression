//
// Libor Novak
// 01/03/2017
//

#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace eic {


/**
 * @brief Kernel that computes the fitness values of all chromozomes in the given population
 *
 * Each block processes one chromozome (individual) from the population. Each block therefore needs canvas
 * for rendering the images in the shared memory. Since the shared memory is quite small one needs to split
 * the whole population into multiple calls of this kernel because all canvas would not fit into the shared
 * memory. Also each chromozome is copied to the shared memory, which needs to be shared between canvas and
 * the chromozome description.
 *
 * This kernel also copies out the rendered images into the global memory g_all_canvas.
 *
 * @param g_target Data from cv::Mat of the target image that we want to compare (w x h x 3)
 * @param g_weights Weights of different parts of the target image (w x h x 1)
 * @param width Width of the target image (w)
 * @param height Height of the target image (h)
 * @param g_population Description vector of the whole population
 * @param offset Id of the first currently processed chromozome (because of the splitting to multiple kernel calls)
 * @param population_size Number of chromozomes in the population
 * @param chromozome_length Number of shapes in each chromozome
 * @param g_out_fitness Output array for the computed fitness values (length = population_size)
 * @param g_all_canvas Array of cv::Mat canvas for rendering of the images
 */
__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int offset, int population_size, int chromozome_length,
                        float *g_out_fitness, int *g_all_canvas);

/**
 * @brief Same as previous kernel, but this one does not copy out the rendered images
 */
__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int offset, int population_size, int chromozome_length,
                        float *g_out_fitness);

}


#endif // KERNELS_H

