#include "kernels.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "components/Config.h"
#include "settings.h"


namespace eic {


namespace {

    /**
     * @brief Set the whole canvas to the initialization value before rendering
     * @param s_canvas
     */
    __device__
    void clearCanvas (int *s_canvas)
    {
        // Make sure we do not delete anything before we are done processing it (this actually happened to me)
        __syncthreads();

        // Number of elements to be processed per thread
        int nept = ceil(float(CANVAS_DIMENSION*CANVAS_DIMENSION) / blockDim.x);

        for (int i = threadIdx.x*nept; i < threadIdx.x*nept+nept; ++i)
        {
            if (i < CANVAS_DIMENSION*CANVAS_DIMENSION)
            {
#ifdef RENDER_AVERAGE
                s_canvas[3*i + 0] = 1;
                s_canvas[3*i + 1] = 1;
                s_canvas[3*i + 2] = 1;
#else
                s_canvas[3*i + 0] = 0;
                s_canvas[3*i + 1] = 0;
                s_canvas[3*i + 2] = 0;
#endif
            }
        }

        __syncthreads();
    }


    /**
     * @brief Renders a transparent circle or its part onto the given canvas
     * @param s_canvas Canvas of size CANVAS_DIMENSION
     * @param canvas_width Width of the canvas in pixels
     * @param canvas_height Height of the canvas in pixels
     * @param r Red channel multiplied already by alpha
     * @param g Green channel multiplied already by alpha
     * @param b Blue channel multiplied already by alpha
     * @param alpha_inv 1-alpha (to avoid the subtraction)
     * @param center_x X coordinate of the circle with respect to canvas
     * @param center_y Y coordinate of the circle with respect to canvas
     * @param radius Radius of the circle
     */
    __device__
    void renderCircle (int *s_canvas, const int canvas_width, const int canvas_height, const int r,
                       const int g, const int b, const float alpha_inv, const int center_x,
                       const int center_y, const int radius)
    {
        // The circle is rendered as follows - a bounding box around the circle is determined and for each
        // pixel inside of the bounding box we determine whether it is inside of the circle or not. If yes,
        // then the color is set accordingly

        const int radius_sq = radius * radius;

        // Determine the bounding box coordinates inside the current canvas
        const int bb_tl_x = max(0, center_x - radius);
        const int bb_tl_y = max(0, center_y - radius);
        const int bb_br_x = min(canvas_width, center_x + radius);
        const int bb_br_y = min(canvas_height, center_y + radius);

        const int bb_width = bb_br_x-bb_tl_x;
        const int bb_height = bb_br_y-bb_tl_y;

        // Traverse the bounding box and render the pixels, which are inside of the circle
        for (int i = threadIdx.x; i < bb_width*bb_height; i += blockDim.x)
        {
            // Get the x and y coordinates of the pixel in the canvas
            int y = int(i / bb_width);
            int x = (i - (y * bb_width));
            x += bb_tl_x; y += bb_tl_y;

            // Check if this point is inside of the circle
            if ((x-center_x)*(x-center_x) + (y-center_y)*(y-center_y) <= radius_sq)
            {
                int pixel_idx = 3*canvas_width*y + 3*x;
                s_canvas[pixel_idx + 0] = alpha_inv*s_canvas[pixel_idx + 0] + r;
                s_canvas[pixel_idx + 1] = alpha_inv*s_canvas[pixel_idx + 1] + g;
                s_canvas[pixel_idx + 2] = alpha_inv*s_canvas[pixel_idx + 2] + b;
            }
        }
    }


    __device__
    void renderCell (int *s_canvas, const int width, const int height, const int tl_x, const int tl_y, const int *g_shape_desc, const int chromozome_length)
    {
        for (int i = 0; i < chromozome_length; ++i)
        {
            // Render each shape
            if (ShapeType(g_shape_desc[0]) == ShapeType::CIRCLE)
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

                // Check if circle intersects current cell - in that case render it
                if (g_shape_desc[5]+g_shape_desc[7] >= tl_x &&
                        g_shape_desc[5]-g_shape_desc[7] < tl_x+width &&
                        g_shape_desc[6]+g_shape_desc[7] >= tl_y &&
                        g_shape_desc[6]-g_shape_desc[7] < tl_y+height)
                {
                    float alpha = float(g_shape_desc[4]) / 100;

                    renderCircle(s_canvas, width, height,
                                 alpha*g_shape_desc[1], alpha*g_shape_desc[2], alpha*g_shape_desc[3],
                                 1-alpha, g_shape_desc[5]-tl_x, g_shape_desc[6]-tl_y, g_shape_desc[7]);
                }
            }

            // Wait for the whole shape to be rendered
            __syncthreads();

            g_shape_desc += DESC_LEN;
        }
    }


    __device__
    void fitnessCell (int *s_canvas, const int canvas_width, const int canvas_height, const int tl_x, const int tl_y, const __uint8_t *g_target, const float *g_weights, const int target_width, const int target_height, float *s_fitness)
    {
        // TODO: Add ROI!

        float my_fitness = 0.0f;

        for (int i = threadIdx.x; i < canvas_width*canvas_height; i += blockDim.x)
        {
            // Get the x and y coordinates of the pixel in the canvas and target
            int y = int(i / canvas_width);
            int x = (i - (y * canvas_width));
            int tx = x + tl_x;
            int ty = y + tl_y;

            float diff = (s_canvas[y*3*canvas_width+3*x]-g_target[ty*3*target_width+3*tx]);
            my_fitness += g_weights[ty*target_width+tx] * diff*diff;
            diff = (s_canvas[y*3*canvas_width+3*x+1]-g_target[ty*3*target_width+3*tx+1]);
            my_fitness += g_weights[ty*target_width+tx] * diff*diff;
            diff = (s_canvas[y*3*canvas_width+3*x+2]-g_target[ty*3*target_width+3*tx+2]);
            my_fitness += g_weights[ty*target_width+tx] * diff*diff;
        }

        atomicAdd(s_fitness, my_fitness);
    }

}



// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int offset, int population_size, int chromozome_length,
                        float *g_out_fitness, int *g_canvas)
{
    // cv::Mat is organized in the h x w x 3 (01c) manner - we want to have the same
    extern __shared__ int s_canvas[];  // size is SHARED_MEM_SIZE (h x w x 3 channels)

    // Variable for keeping the intermediate fitness value
    __shared__ float s_fitness[1];
    if (threadIdx.x == 0) s_fitness[0] = 0.0f;


    // Chromozome id that is being rendered is given by the block id
    unsigned int ch_id = offset + blockIdx.x;

    // Plot each shape in the chromozome
    int *g_chromozome = g_population + ch_id*(chromozome_length*DESC_LEN+5);
    int *g_shape_desc = g_chromozome + 5;


    // Split the rendering to a grid of cells of size CANVAS_DIMENSION x CANVAS_DIMENSION
    // Because the whole image does not fit into the shared memory we have to render it in pieces
    int cols = ceil(float(width) / CANVAS_DIMENSION);
    int rows = ceil(float(height) / CANVAS_DIMENSION);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Render this cell
            int tl_x = j*CANVAS_DIMENSION;
            int tl_y = i*CANVAS_DIMENSION;
            int br_x = min(tl_x+CANVAS_DIMENSION, width);
            int br_y = min(tl_y+CANVAS_DIMENSION, height);
            int cell_width = br_x-tl_x;
            int cell_height = br_y-tl_y;

            clearCanvas(s_canvas);

            renderCell(s_canvas, cell_width, cell_height, tl_x, tl_y, g_shape_desc, chromozome_length);

            // Compute fitness on this part of the image
            fitnessCell(s_canvas, cell_width, cell_height, tl_x, tl_y, g_target, g_weights, width, height, s_fitness);

//            // Copy the rendered part to the output
//            for (int k = threadIdx.x; k < cell_width*cell_height; k += blockDim.x)
//            {
//                int row = k / cell_width;
//                int col = k - row*cell_width;
//                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 0] = s_canvas[3*row*cell_width + 3*col + 0];
//                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 1] = s_canvas[3*row*cell_width + 3*col + 1];
//                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 2] = s_canvas[3*row*cell_width + 3*col + 2];
//            }
        }
    }


    __syncthreads();

    // Copy the final fitness value to the output
    if (threadIdx.x == 0)
    {
        g_out_fitness[ch_id] = s_fitness[0];
    }
}


}
