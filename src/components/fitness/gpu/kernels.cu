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
                s_canvas[4*i + 0] = 1;
                s_canvas[4*i + 1] = 1;
                s_canvas[4*i + 2] = 1;
                s_canvas[4*i + 3] = 1;
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
     * @brief Copy the whole chromozome (only the shape description part) to the shared memory
     */
    __device__
    void copyChromozome (int *g_shape_desc, int *s_shape_desc, int chromozome_length)
    {
        // Number of elements to be processed per thread
        int nept = ceil(float(chromozome_length*DESC_LEN) / blockDim.x);

        for (int i = threadIdx.x*nept; i < threadIdx.x*nept+nept; ++i)
        {
            // Copy the value over
            if (i < chromozome_length*DESC_LEN) s_shape_desc[i] = g_shape_desc[i];
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
#ifdef RENDER_AVERAGE
    void renderCircle (int *s_canvas, const int canvas_width, const int canvas_height, const int r,
                       const int g, const int b, const int a, const int center_x,
                       const int center_y, const int radius)
#else
    void renderCircle (int *s_canvas, const int canvas_width, const int canvas_height, const int r,
                       const int g, const int b, const float alpha_inv, const int center_x,
                       const int center_y, const int radius)
#endif
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

        // Number of elements to be processed per thread
        int nept = ceil(float(bb_width*bb_height) / blockDim.x);

        // Traverse the bounding box and render the pixels, which are inside of the circle
        for (int i = threadIdx.x*nept; i < threadIdx.x*nept+nept; ++i)
        {
            if (i < bb_width*bb_height)
            {
                // Get the x and y coordinates of the pixel in the canvas
                int y = int(i / bb_width);
                int x = (i - (y * bb_width));
                x += bb_tl_x; y += bb_tl_y;

                // Check if this point is inside of the circle
                if ((x-center_x)*(x-center_x) + (y-center_y)*(y-center_y) <= radius_sq)
                {
#ifdef RENDER_AVERAGE
                    int pixel_idx = 4*canvas_width*y + 4*x;
                    s_canvas[pixel_idx + 0] = s_canvas[pixel_idx + 0] + r;
                    s_canvas[pixel_idx + 1] = s_canvas[pixel_idx + 1] + g;
                    s_canvas[pixel_idx + 2] = s_canvas[pixel_idx + 2] + b;
                    s_canvas[pixel_idx + 3] = s_canvas[pixel_idx + 3] + a;
#else
                    int pixel_idx = 3*canvas_width*y + 3*x;
                    s_canvas[pixel_idx + 0] = alpha_inv*s_canvas[pixel_idx + 0] + r;
                    s_canvas[pixel_idx + 1] = alpha_inv*s_canvas[pixel_idx + 1] + g;
                    s_canvas[pixel_idx + 2] = alpha_inv*s_canvas[pixel_idx + 2] + b;
#endif
                }
            }
        }
    }


    /**
     * @brief Renderes all shapes into the given grid cell
     * @param s_canvas Canvas of the grid cell
     * @param width Width of the grid cell
     * @param height Height of the grid cell
     * @param tl_x Top left x coordinate of the grid cell in the whole image
     * @param tl_y Top left y coordinate of the grid cell in the whole image
     * @param s_shape_desc Pointer to the description of shapes in the chromozome
     * @param chromozome_length Length of the chromozome
     */
    __device__
    void renderCell (int *s_canvas, const int width, const int height, const int tl_x, const int tl_y,
                     const int *s_shape_desc, const int chromozome_length)
    {
        for (int i = 0; i < chromozome_length; ++i)
        {
            // Render each shape
            if (ShapeType(s_shape_desc[0]) == ShapeType::CIRCLE)
            {
                // Circle has the following representation:
                // [0] = ShapeType::CIRCLE, [1] = R, [2] = G, [3] = B, [4] = alpha, [5] = center.x,
                // [6] = center.y, [7] = radius

                // Check if circle intersects current cell - in that case render it
                if (s_shape_desc[5]+s_shape_desc[7] >= tl_x &&
                        s_shape_desc[5]-s_shape_desc[7] < tl_x+width &&
                        s_shape_desc[6]+s_shape_desc[7] >= tl_y &&
                        s_shape_desc[6]-s_shape_desc[7] < tl_y+height)
                {
                    float alpha = float(s_shape_desc[4]) / 100;

#ifdef RENDER_AVERAGE
                    renderCircle(s_canvas, width, height,
                                 alpha*s_shape_desc[1], alpha*s_shape_desc[2], alpha*s_shape_desc[3],
                                 s_shape_desc[4], s_shape_desc[5]-tl_x, s_shape_desc[6]-tl_y, s_shape_desc[7]);
#else
                    renderCircle(s_canvas, width, height,
                                 alpha*s_shape_desc[1], alpha*s_shape_desc[2], alpha*s_shape_desc[3],
                                 1-alpha, s_shape_desc[5]-tl_x, s_shape_desc[6]-tl_y, s_shape_desc[7]);
#endif
                }
            }

            // Wait for the whole shape to be rendered
            __syncthreads();

            s_shape_desc += DESC_LEN;
        }
    }


    /**
     * @brief Compute fitness function sum in the current (given) grid cell
     * @param s_canvas Canvas of the current grid cell with plotted shapes
     * @param cell_width
     * @param cell_height
     * @param tl_x
     * @param tl_y
     * @param g_target Target image matrix
     * @param g_weights Matrix of weights
     * @param target_width
     * @param target_height
     * @param s_roi Description of the region of interest (has 5 values)
     * @param s_fitness Output variable for the fitness
     */
    __device__
    void fitnessCell (int *s_canvas, const int cell_width, const int cell_height, const int tl_x,
                      const int tl_y, const __uint8_t *g_target, const float *g_weights,
                      const int target_width, const int target_height, int *s_roi, float *s_fitness)
    {
        float my_fitness = 0.0f;

        for (int i = threadIdx.x; i < cell_width*cell_height; i += blockDim.x)
        {
            // Get the x and y coordinates of the pixel in the canvas and target
            int y = int(i / cell_width);
            int x = (i - (y * cell_width));
            int tx = x + tl_x;
            int ty = y + tl_y;

            float weight = g_weights[ty*target_width+tx];

            // Check the ROI
            if (s_roi[0] == 1 && tx >= s_roi[1] && tx < s_roi[3] && ty >= s_roi[2] && ty < s_roi[4])
            {
                // Pixel inside ROI
                weight *= s_roi[5];
            }

#ifdef RENDER_AVERAGE
            // Divide the sums by the sum of the alpha channel
            float sum_alpha = float(s_canvas[y*4*cell_width+4*x+3]) / 100.0f;
            s_canvas[y*4*cell_width+4*x]   /= sum_alpha;
            s_canvas[y*4*cell_width+4*x+1] /= sum_alpha;
            s_canvas[y*4*cell_width+4*x+2] /= sum_alpha;

            float diff = (s_canvas[y*4*cell_width+4*x]-g_target[ty*3*target_width+3*tx]);
            my_fitness += weight * diff*diff;
            diff = (s_canvas[y*4*cell_width+4*x+1]-g_target[ty*3*target_width+3*tx+1]);
            my_fitness += weight * diff*diff;
            diff = (s_canvas[y*4*cell_width+4*x+2]-g_target[ty*3*target_width+3*tx+2]);
            my_fitness += weight * diff*diff;
#else
            float diff = (s_canvas[y*3*cell_width+3*x]-g_target[ty*3*target_width+3*tx]);
            my_fitness += weight * diff*diff;
            diff = (s_canvas[y*3*cell_width+3*x+1]-g_target[ty*3*target_width+3*tx+1]);
            my_fitness += weight * diff*diff;
            diff = (s_canvas[y*3*cell_width+3*x+2]-g_target[ty*3*target_width+3*tx+2]);
            my_fitness += weight * diff*diff;
#endif
        }

        atomicAdd(s_fitness, my_fitness);
    }


    /**
     * @brief Copies the current (given) cell to the output canvas
     * @param s_canvas Current rendered grid cell
     * @param cell_width
     * @param cell_height
     * @param tl_x
     * @param tl_y
     * @param g_canvas Output canvas
     * @param target_width Width of the output canvas (same as target width)
     * @param target_height Height of the output canvas (same as target height)
     */
    __device__
    void copyCell (int *s_canvas, const int cell_width, const int cell_height, const int tl_x, const int tl_y,
                   int *g_canvas, const int target_width, const int target_height)
    {
        // Number of elements to be processed per thread
        int nept = ceil(float(cell_width*cell_height) / blockDim.x);

        for (int i = threadIdx.x*nept; i < threadIdx.x*nept+nept; ++i)
        {
            if (i < cell_width*cell_height)
            {
                int row = i / cell_width;
                int col = i - row*cell_width;

                // Copy RGB channels on the pixel
#ifdef RENDER_AVERAGE
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 0] = s_canvas[4*row*cell_width + 4*col + 0];
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 1] = s_canvas[4*row*cell_width + 4*col + 1];
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 2] = s_canvas[4*row*cell_width + 4*col + 2];
#else
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 0] = s_canvas[3*row*cell_width + 3*col + 0];
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 1] = s_canvas[3*row*cell_width + 3*col + 1];
                g_canvas[3*(tl_y+row)*target_width + 3*(tl_x+col) + 2] = s_canvas[3*row*cell_width + 3*col + 2];
#endif
            }
        }
    }

}



// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int population_size, int chromozome_length, float *g_out_fitness, int *g_all_canvas)
{
    // VERSION THAT WRITES OUT THE RENDERED IMAGES

    // cv::Mat is organized in the h x w x 3 (01c) manner - we want to have the same
    extern __shared__ int s_canvas[];  // size is SHARED_MEM_SIZE (h x w x 3 channels)

    // Variable for keeping the intermediate fitness value
    __shared__ float s_fitness[1];
    if (threadIdx.x == 0) s_fitness[0] = 0.0f;

    // Chromozome id that is being rendered is given by the block id
    unsigned int ch_id = blockIdx.x;

    // Get the pointer to the memory, where the chromozome that is being processed by this block is
    int *g_chromozome = g_population + ch_id*(chromozome_length*DESC_LEN+6);
    int *g_shape_desc = g_chromozome + 6;  // First 6 numbers are ROI

    // Region of interest - we do not want to read it from the global memory all the time
    __shared__ int s_roi[6];
    if (threadIdx.x < 6) s_roi[threadIdx.x] = g_chromozome[threadIdx.x];

    // Copy the chromozome into the shared memory - we want faster access during rendering. Since we will be
    // reading the chromozome multiple times this should pay off
    __shared__ int s_shape_desc[CHROMOZOME_MEM_SIZE/sizeof(int)];
    copyChromozome(g_shape_desc, s_shape_desc, chromozome_length);

    // Get the pointer to the canvas, which corresponds to the currently rendered chromozome
    int *g_canvas = g_all_canvas + (3*width*height * ch_id);


    // Split the rendering to a grid of cells of size CANVAS_DIMENSION x CANVAS_DIMENSION
    // We have to do this because the whole image does not fit into the shared memory - we have to render it
    // in pieces, which can fit
    int cols = ceil(float(width) / CANVAS_DIMENSION);
    int rows = ceil(float(height) / CANVAS_DIMENSION);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Render this cell
            int tl_x = j*CANVAS_DIMENSION;
            int tl_y = i*CANVAS_DIMENSION;
            int cell_width = min(tl_x+CANVAS_DIMENSION, width)-tl_x;
            int cell_height = min(tl_y+CANVAS_DIMENSION, height)-tl_y;

            // Set the canvas to 0 - remove previously rendered shapes
            clearCanvas(s_canvas);

            // Render this cell into the canvas
            renderCell(s_canvas, cell_width, cell_height, tl_x, tl_y, s_shape_desc, chromozome_length);

            // Compute fitness on this part of the image
            fitnessCell(s_canvas, cell_width, cell_height, tl_x, tl_y, g_target, g_weights, width, height,
                        s_roi, s_fitness);

            // Copy the rendered image to the global memory - the output
            copyCell(s_canvas, cell_width, cell_height, tl_x, tl_y, g_canvas, width, height);
        }
    }


    __syncthreads();

    // Copy the final fitness value to the output
    if (threadIdx.x == 0)
    {
        g_out_fitness[ch_id] = s_fitness[0];
    }
}


__global__
void populationFitness (__uint8_t *g_target, float *g_weights, int width, int height, int *g_population,
                        int population_size, int chromozome_length, float *g_out_fitness)
{
    // VERSION THAT DOES NOT WRITE OUT THE RENDERED IMAGES -> MUCH FASTER!

    // cv::Mat is organized in the h x w x 3 (01c) manner - we want to have the same
    extern __shared__ int s_canvas[];  // size is SHARED_MEM_SIZE (h x w x 3 channels)

    // Variable for keeping the intermediate fitness value
    __shared__ float s_fitness[1];
    if (threadIdx.x == 0) s_fitness[0] = 0.0f;

    // Chromozome id that is being rendered is given by the block id
    unsigned int ch_id = blockIdx.x;

    // Get the pointer to the memory, where the chromozome that is being processed by this block is
    int *g_chromozome = g_population + ch_id*(chromozome_length*DESC_LEN+6);
    int *g_shape_desc = g_chromozome + 6;  // First 6 numbers are ROI

    // Region of interest - we do not want to read it from the global memory all the time
    __shared__ int s_roi[6];
    if (threadIdx.x < 6) s_roi[threadIdx.x] = g_chromozome[threadIdx.x];

    // Copy the chromozome into the shared memory - we want faster access during rendering. Since we will be
    // reading the chromozome multiple times this should pay off
    __shared__ int s_shape_desc[CHROMOZOME_MEM_SIZE/sizeof(int)];
    copyChromozome(g_shape_desc, s_shape_desc, chromozome_length);


    // Split the rendering to a grid of cells of size CANVAS_DIMENSION x CANVAS_DIMENSION
    // We have to do this because the whole image does not fit into the shared memory - we have to render it
    // in pieces, which can fit
    int cols = ceil(float(width) / CANVAS_DIMENSION);
    int rows = ceil(float(height) / CANVAS_DIMENSION);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // Render this cell
            int tl_x = j*CANVAS_DIMENSION;  // top left x
            int tl_y = i*CANVAS_DIMENSION;  // top left y
            int cell_width = min(tl_x+CANVAS_DIMENSION, width)-tl_x;
            int cell_height = min(tl_y+CANVAS_DIMENSION, height)-tl_y;

            // Set the canvas to 0 - remove previously rendered shapes
            clearCanvas(s_canvas);

            // Render this cell into the canvas
            renderCell(s_canvas, cell_width, cell_height, tl_x, tl_y, s_shape_desc, chromozome_length);

            // Compute fitness on this part of the image
            fitnessCell(s_canvas, cell_width, cell_height, tl_x, tl_y, g_target, g_weights, width, height,
                        s_roi, s_fitness);
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
