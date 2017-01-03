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


    __device__
    void renderCircle (int *s_canvas, const int canvas_width, const int canvas_height, const int r,
                       const int g, const int b, const float alpha_inv, const int center_x,
                       const int center_y, const int radius)
    {
        // cv::Mat is organized in the h x w x 3 (01c) manner - we want to have the same
        int diameter = 2 * radius;
        int radius_sq = radius * radius;
        int tl_x   = center_x - radius;
        int tl_y   = center_y - radius;

        for (int i = threadIdx.x; i < diameter*diameter; i += blockDim.x)
        {
            int y = int(i / diameter);
            int x = (i - (y * diameter));
            x += tl_x; y += tl_y;

            if ((x-center_x)*(x-center_x) + (y-center_y)*(y-center_y) < radius_sq &&  // Point inside circle
                    x >= 0 && y >= 0 && x < canvas_width && y < canvas_height)        // Image bounds
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

}


// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void populationFitness (__uint8_t *g_target, unsigned int width, unsigned int height, int *g_population,
                        unsigned int population_size, unsigned int chromozome_length, float *g_out_fitness,
                        int * g_canvas)
{
//    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ int s_canvas[];  // size is SHARED_MEM_SIZE



    // Chromozome id that is being rendered is given by the block id
    unsigned int ch_id = blockIdx.x;

    // Plot each shape in the chromozome
    int *g_chromozome = g_population + ch_id*(chromozome_length*DESC_LEN+1);
    int *g_shape_desc = g_chromozome + 1;


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

            // Copy the rendered part to the output
            for (int k = threadIdx.x; k < cell_width*cell_height; k += blockDim.x)
            {
                int row = k / cell_width;
                int col = k - row*cell_width;
                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 0] = s_canvas[3*row*cell_width + 3*col + 0];
                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 1] = s_canvas[3*row*cell_width + 3*col + 1];
                g_canvas[3*(tl_y+row)*width + 3*(tl_x+col) + 2] = s_canvas[3*row*cell_width + 3*col + 2];
            }
        }
    }





//    for (int i = 0; i < chromozome_length; ++i)
//    {
//        // Render each shape
//        if (ShapeType(g_shape_desc[0]) == ShapeType::CIRCLE)
//        {
//            renderCircle(s_canvas, width, height, g_shape_desc);
//        }

//        __syncthreads();

//        g_shape_desc += DESC_LEN;
//    }

    // Compute fitness


//    for (int i = threadIdx.x; i < width*height; i += blockDim.x)
//    {
//        int row = i / width;
//        int col = i - row*width;
//        g_canvas[3*row*width + 3*col + 0] = s_canvas[3*row*width + 3*col + 0];
//        g_canvas[3*row*width + 3*col + 1] = s_canvas[3*row*width + 3*col + 1];
//        g_canvas[3*row*width + 3*col + 2] = s_canvas[3*row*width + 3*col + 2];
//    }
}


}
