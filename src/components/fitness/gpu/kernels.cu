#include "kernels.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "components/Config.h"

namespace eic {


namespace {

    __device__
    void renderCircle (float *s_canvas, const unsigned int width, const unsigned int height, float *g_shape_desc)
    {
        // Circle has the following representation
        // [0] = ShapeType::CIRCLE
        // [1] = R
        // [2] = G
        // [3] = B
        // [4] = alpha
        // [5] = center.x
        // [6] = center.y
        // [7] = radius

        // cv::Mat is organized in the h x w x 3 (01c) manner - we want to have the same
        int radius = g_shape_desc[7];
        int diameter = 2 * radius;
        int tl_x   = g_shape_desc[5] - radius;
        int tl_y   = g_shape_desc[6] - radius;

        for (int i = threadIdx.x; i < diameter*diameter; i += blockDim.x)
        {
            int y = int(i / diameter);
            int x = (i - (y * diameter));
            x += tl_x; y += tl_y;

            // Check the image bounds
            if (x >= 0 && y >= 0 && x < width && y < height)
            {
                s_canvas[3*width*y + 3*x + 0] = g_shape_desc[1]; // R
                s_canvas[3*width*y + 3*x + 1] = g_shape_desc[2]; // G
                s_canvas[3*width*y + 3*x + 2] = g_shape_desc[3]; // B
            }
        }
    }

}


// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void populationFitness (__uint8_t *g_target, unsigned int width, unsigned int height, float *g_population,
                        unsigned int population_size, unsigned int chromozome_length, float *g_out_fitness,
                        float * g_canvas)
{
//    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ float s_canvas[];  // width x height x 3 channels
    // Clear the whole canvas - set to 0
    for (int i = threadIdx.x; i < width*height; i += blockDim.x)
    {
        int row = i / width;
        int col = i - row*width;
        s_canvas[3*row*width + 3*col + 0] = 0;
        s_canvas[3*row*width + 3*col + 1] = 0;
        s_canvas[3*row*width + 3*col + 2] = 0;
    }

    // Chromozome id that is being rendered is given by the block id
    unsigned int ch_id = blockIdx.x;

    // Plot each shape in the chromozome
    float *g_chromozome = g_population + ch_id*(chromozome_length*DESC_LEN+1);
    float *g_shape_desc = g_chromozome + 1;
    for (int i = 0; i < chromozome_length; ++i)
    {
        // Render each shape
        if (ShapeType(g_shape_desc[0]) == ShapeType::CIRCLE)
        {
            renderCircle(s_canvas, width, height, g_shape_desc);
        }

        __syncthreads();

        g_shape_desc += DESC_LEN;
    }

    // Compute fitness


    for (int i = threadIdx.x; i < width*height; i += blockDim.x)
    {
        int row = i / width;
        int col = i - row*width;
        g_canvas[3*row*width + 3*col + 0] = s_canvas[3*row*width + 3*col + 0];
        g_canvas[3*row*width + 3*col + 1] = s_canvas[3*row*width + 3*col + 1];
        g_canvas[3*row*width + 3*col + 2] = s_canvas[3*row*width + 3*col + 2];
    }
}


}
