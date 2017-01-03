#include "GPUFitness.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "components/Chromozome.h"
#include "check_error.h"
#include "kernels.h"
#include "settings.h"


namespace eic {

namespace {

    /**
     * @brief Determines the number of multiprocessors of the current GPU
     */
    int getNumMultiprocessors ()
    {
        cudaDeviceProp device_properties;
        CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

        return device_properties.multiProcessorCount;
    }

}


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{
    // CAREFUL! For this fitness computation to work each chromozome must have the same length and
    // the same target!

    assert(chromozomes.size() > 0);

    // Create population description from the chromozome list - each chromozome is encoded into an array
    // of integers
    int population_size    = chromozomes.size();
    int chromozome_length  = chromozomes[0]->size();
    int description_length = population_size * (5 + chromozome_length*DESC_LEN);  // The 5 is for fitness ROI

    std::vector<int> population(description_length, 0);
    for (int i = 0; i < population_size; ++i)
    {
        int population_idx = i * (5 + chromozome_length*DESC_LEN);

        if (chromozomes[i]->_roi_active)
        {
            // Write and activate the ROI for this chromozome
            population[population_idx] = 1;
            population[population_idx + 1] = chromozomes[i]->_roi.x;
            population[population_idx + 2] = chromozomes[i]->_roi.y;
            population[population_idx + 3] = chromozomes[i]->_roi.width;
            population[population_idx + 4] = chromozomes[i]->_roi.height;
        }

        // Write all shapes
        for (int j = 0; j < chromozome_length; j++)
        {
            int population_shape_idx = population_idx + 5 + (chromozome_length-j-1)*DESC_LEN;
            chromozomes[i]->operator[](j)->writeDescription(&(population[population_shape_idx]));
        }
    }

    // Copy the population description to GPU
    int *g_population; cudaMalloc((void**)&g_population, description_length*sizeof(int));
    CHECK_ERROR(cudaMemcpy(g_population, population.data(), description_length*sizeof(int), cudaMemcpyHostToDevice));


    // Copy the target to GPU
    cv::Mat target = chromozomes[0]->getTarget()->blurred_image;
    int target_size = 3*target.rows*target.cols;
    uchar *g_target; cudaMalloc((void**)&g_target, target_size*sizeof(uchar));
    CHECK_ERROR(cudaMemcpy(g_target, target.ptr<uchar>(), target_size*sizeof(uchar), cudaMemcpyHostToDevice));
    // Copy the weights to GPU
    cv::Mat weights = chromozomes[0]->getTarget()->weights;
    float *g_weights; cudaMalloc((void**)&g_weights, target_size/3*sizeof(float));
    CHECK_ERROR(cudaMemcpy(g_weights, weights.ptr<float>(), target_size/3*sizeof(float), cudaMemcpyHostToDevice));

    // Allocate memory for output fitness values
    float *g_out_fitness; cudaMalloc((void**)&g_out_fitness, population_size*sizeof(float));


    int* g_canvas; cudaMalloc((void**)&g_canvas, target_size*sizeof(int));


    // Each rendering can run only on one multiprocessor!!! Because of the shared memory - we have to split
    // the whole population to several kernel calls
    int num_multiprocessors = getNumMultiprocessors();
    int num_iterations = ceil(double(population_size) / num_multiprocessors);
    for (int i = 0; i < num_iterations; ++i)
    {
        int offset = i * num_multiprocessors;
        int end    = min(offset+num_multiprocessors, population_size);

        int num_blocks = end-offset;
        populationFitness<<< num_blocks, THREADS_PER_BLOCK, SHARED_MEM_SIZE >>>(
            g_target, g_weights, target.cols, target.rows, g_population, offset, population_size,
            chromozome_length, g_out_fitness, g_canvas
        );
    }



//    cv::Mat canvas(target.size(), CV_32SC3);
//    CHECK_ERROR(cudaMemcpy(canvas.ptr<int>(), g_canvas, target_size*sizeof(int), cudaMemcpyDeviceToHost));

////    std::cout << canvas << std::endl;

//    canvas.convertTo(canvas, CV_8UC3);
//    cv::cvtColor(canvas, canvas, CV_RGB2BGR);
//    cv::imwrite("render_gpu.png", canvas);

    float out_fitness[population_size];
    CHECK_ERROR(cudaMemcpy(out_fitness, g_out_fitness, population_size*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < population_size; ++i)
    {
        chromozomes[i]->_fitness = out_fitness[i];
        chromozomes[i]->_dirty = false;
        std::cout << "[" << i << "] GPU fitness: " << out_fitness[i] << std::endl;
    }


    cudaFree(g_target);
    cudaFree(g_weights);
    cudaFree(g_out_fitness);
    cudaFree(g_population);
    cudaFree(g_canvas);

//    exit(EXIT_SUCCESS);
}


void computeFitnessGPU (const std::shared_ptr<Chromozome> &ch, bool write_channels)
{
    std::vector<std::shared_ptr<Chromozome>> chromozomes;
    chromozomes.push_back(ch);

    computeFitnessGPU(chromozomes, write_channels);
}


bool initializeGPU ()
{
    // Find out if there is a CUDA capable device
    int device_count;
    CHECK_ERROR(cudaGetDeviceCount(&device_count));

    // Get properties of the device
    cudaDeviceProp device_properties;
    CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

    if (device_count == 0 || (device_properties.major == 0 && device_properties.minor == 0))
    {
        // Error, we cannot initialize
        return false;
    }
    else
    {
        // Copying a dummy to the device will initialize it
        int* gpu_dummy;
        cudaMalloc((void**)&gpu_dummy, sizeof(int));
        cudaFree(gpu_dummy);

        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "Device name:                    " << device_properties.name << std::endl;
        std::cout << "Compute capability:             " << device_properties.major << "." << device_properties.minor << std::endl;
        std::cout << "Total global memory:            " << device_properties.totalGlobalMem << std::endl;
        std::cout << "Multiprocessor count:           " << device_properties.multiProcessorCount << std::endl;
        std::cout << "Max threads per block:          " << device_properties.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim:                " << device_properties.maxThreadsDim[0] << std::endl;
        std::cout << "Max grid size:                  " << device_properties.maxGridSize[0] << std::endl;
        std::cout << "Shared mem per block:           " << device_properties.sharedMemPerBlock << std::endl;
        std::cout << "Shared mem per multiprocessor:  " << device_properties.sharedMemPerMultiprocessor << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        return true;
    }
}


}

