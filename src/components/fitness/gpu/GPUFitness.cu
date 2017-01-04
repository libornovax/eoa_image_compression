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
     * @brief Determines the number of concurrent blocks we can launch on the GPU
     */
    int getNumConcurentBlocks ()
    {
        static int num_concurrent_blocks = 0;

        if (num_concurrent_blocks == 0)
        {
            // The number of block we can launch at once is restricted by the memory requirements of the block
            cudaDeviceProp device_properties;
            CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

            // One multiprocessor sometimes has enough memory to accomodate multiple blocks
            int mp_blocks = device_properties.sharedMemPerMultiprocessor / device_properties.sharedMemPerBlock;
            num_concurrent_blocks = device_properties.multiProcessorCount * mp_blocks;
        }

        return num_concurrent_blocks;
    }


    /**
     * @brief Create the description vector of the whole population (will be then copied to GPU)
     * @param chromozomes Vector of chromozomes in the population
     * @param description_length The total length of the population description vector
     * @param chromozome_length Number of shapes in one chromozome
     */
    std::vector<int> createPopulationDescription (const std::vector<std::shared_ptr<Chromozome>> &chromozomes,
                                                  int description_length, int chromozome_length)
    {
        std::vector<int> population(description_length, 0);

        for (int i = 0; i < chromozomes.size(); ++i)
        {
            // Index of the first element of this chromozome in the whole population vector
            int population_idx = i * (6 + chromozome_length*DESC_LEN);

            if (chromozomes[i]->roiActive())
            {
                // Write and activate the ROI for this chromozome
                population[population_idx] = 1;
                population[population_idx + 1] = chromozomes[i]->getROI().x;
                population[population_idx + 2] = chromozomes[i]->getROI().y;
                population[population_idx + 3] = chromozomes[i]->getROI().x + chromozomes[i]->getROI().width;
                population[population_idx + 4] = chromozomes[i]->getROI().y + chromozomes[i]->getROI().height;
                population[population_idx + 5] = 1 + chromozomes[i]->getTarget()->image_size.area()
                        / chromozomes[i]->getROI().area();
            }

            // Write all shapes
            for (int j = 0; j < chromozome_length; j++)
            {
                int population_shape_idx = population_idx + 6 + (chromozome_length-j-1)*DESC_LEN;
                chromozomes[i]->operator[](j)->writeDescription(&(population[population_shape_idx]));
            }
        }

        return population;
    }


    /**
     * @brief Launches a fitness computing kernel, which also copies out the rendered images
     * This kernel is slower because it must copy all images out of the GPU, but we have no choice if we want
     * to display the rendered images. Therefore this function should not be called too often
     */
    void computeAndWriteChannels (int population_size, int chromozome_length, const cv::Mat &target,
                                  uchar *g_target, float *g_weights, int *g_population, float *g_out_fitness,
                                  const std::vector<std::shared_ptr<Chromozome>> &chromozomes)
    {
        // Allocate memory on GPU for the output images
        int target_size = 3*target.rows*target.cols;
        int* g_all_canvas; cudaMalloc((void**)&g_all_canvas, population_size*target_size*sizeof(int));

        // -- FITNESS COMPUTING KERNEL -- //
        // Each rendering can run only on one multiprocessor because of the shared memory size - we have
        // to split the whole population into several kernel calls
        int num_concurrent_blocks = getNumConcurentBlocks();
        int num_iterations = ceil(double(population_size) / num_concurrent_blocks);

        for (int i = 0; i < num_iterations; ++i)
        {
            int offset     = i * num_concurrent_blocks;
            int end        = min(offset+num_concurrent_blocks, population_size);
            int num_blocks = end - offset;

            populationFitness<<< num_blocks, THREADS_PER_BLOCK, CANVAS_MEM_SIZE >>>(
                g_target, g_weights, target.cols, target.rows, g_population, offset, population_size,
                chromozome_length, g_out_fitness, g_all_canvas
            );
        }

        for (int i = 0; i < population_size; ++i)
        {
            // Copy the rendered channels from the GPU
            cv::Mat image(target.size(), CV_32SC3);
            CHECK_ERROR(cudaMemcpy(image.ptr<int>(), g_all_canvas+i*target_size, target_size*sizeof(int),
                                   cudaMemcpyDeviceToHost));

            image.convertTo(image, CV_8UC3);
            cv::split(image, chromozomes[i]->channels());
        }

        cudaFree(g_all_canvas);
    }


    /**
     * @brief Calls a fitness computing kernel, which only computes fitness on the GPU
     * The kernel called from here does not copy any data from the gpu and does not keep the rendered images
     * in the GPU memory
     */
    void computeOnly (int population_size, int chromozome_length, const cv::Mat &target, uchar *g_target,
                      float *g_weights, int *g_population, float *g_out_fitness)
    {
        // -- FITNESS COMPUTING KERNEL -- //
        // Each rendering can run only on one multiprocessor because of the shared memory size - we have
        // to split the whole population into several kernel calls
        int num_concurrent_blocks = getNumConcurentBlocks();
        int num_iterations = ceil(double(population_size) / num_concurrent_blocks);

        for (int i = 0; i < num_iterations; ++i)
        {
            int offset     = i * num_concurrent_blocks;
            int end        = min(offset+num_concurrent_blocks, population_size);
            int num_blocks = end - offset;

            populationFitness<<< num_blocks, THREADS_PER_BLOCK, CANVAS_MEM_SIZE >>>(
                g_target, g_weights, target.cols, target.rows, g_population, offset, population_size,
                chromozome_length, g_out_fitness
            );
        }
    }
}


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{
    // CAREFUL! For this fitness computation to work each chromozome must have the same length and
    // the same target!

    assert(chromozomes.size() > 0);

    int population_size    = chromozomes.size();
    int chromozome_length  = chromozomes[0]->size();
    int description_length = population_size * (6 + chromozome_length*DESC_LEN);  // The 6 is for fitness ROI

    assert(CHROMOZOME_MEM_SIZE >= chromozome_length*DESC_LEN);


    // Create population description from the chromozome list
    // Each chromozome is encoded into an array of integers, which are appended together to create one long
    // population description vector, which can then be easily transfered to GPU
    std::vector<int> population = createPopulationDescription(chromozomes, description_length,
                                                              chromozome_length);
    // Copy the population description to GPU
    int *g_population; cudaMalloc((void**)&g_population, description_length*sizeof(int));
    CHECK_ERROR(cudaMemcpy(g_population, population.data(), description_length*sizeof(int),
                           cudaMemcpyHostToDevice));

    // Copy the target to GPU
    cv::Mat target  = chromozomes[0]->getTarget()->blurred_image;
    int target_size = 3*target.rows*target.cols;
    uchar *g_target; cudaMalloc((void**)&g_target, target_size*sizeof(uchar));
    CHECK_ERROR(cudaMemcpy(g_target, target.ptr<uchar>(), target_size*sizeof(uchar), cudaMemcpyHostToDevice));

    // Copy the weights to GPU
    float *g_weights; cudaMalloc((void**)&g_weights, target_size/3*sizeof(float));
    CHECK_ERROR(cudaMemcpy(g_weights, chromozomes[0]->getTarget()->weights.ptr<float>(),
                           target_size/3*sizeof(float), cudaMemcpyHostToDevice));

    // Allocate memory for output fitness values
    float *g_out_fitness; cudaMalloc((void**)&g_out_fitness, population_size*sizeof(float));


    // -- LAUNCH THE CUDA KERNELS -- //
    if (write_channels)
    {
        // Launch a kernel, which copies out the rendered images
        computeAndWriteChannels(population_size, chromozome_length, target, g_target, g_weights, g_population,
                                g_out_fitness, chromozomes);
    }
    else
    {
        // Launch a kernel, which only computes fitness
        computeOnly(population_size, chromozome_length, target, g_target, g_weights, g_population,
                    g_out_fitness);
    }


    // Copy the fitness values from the GPU
    float out_fitness[population_size];
    CHECK_ERROR(cudaMemcpy(out_fitness, g_out_fitness, population_size*sizeof(float), cudaMemcpyDeviceToHost));
    // Copy the fitness values into the chromozomes
    for (int i = 0; i < population_size; ++i)
    {
        chromozomes[i]->_fitness = out_fitness[i];
        chromozomes[i]->_dirty = false;
    }

    cudaFree(g_target);
    cudaFree(g_weights);
    cudaFree(g_out_fitness);
    cudaFree(g_population);
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

        std::cout << "=================================== GPU ====================================" << std::endl;
        std::cout << "Device name:                    " << device_properties.name << std::endl;
        std::cout << "Compute capability:             " << device_properties.major << "." << device_properties.minor << std::endl;
        std::cout << "Total global memory:            " << device_properties.totalGlobalMem << std::endl;
        std::cout << "Multiprocessor count:           " << device_properties.multiProcessorCount << std::endl;
        std::cout << "Max threads per block:          " << device_properties.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim:                " << device_properties.maxThreadsDim[0] << std::endl;
        std::cout << "Max grid size:                  " << device_properties.maxGridSize[0] << std::endl;
        std::cout << "Shared mem per block:           " << device_properties.sharedMemPerBlock << std::endl;
        std::cout << "Shared mem per multiprocessor:  " << device_properties.sharedMemPerMultiprocessor << std::endl;
        std::cout << "============================================================================" << std::endl;
        std::cout << "Num concurrent blocks:          " << getNumConcurentBlocks() << std::endl;
        std::cout << "THREADS PER BLOCK:              " << THREADS_PER_BLOCK << std::endl;
        std::cout << "CANVAS MEMORY SIZE:             " << CANVAS_MEM_SIZE << std::endl;
        std::cout << "CHROMOZOME MEMORY SIZE:         " << CHROMOZOME_MEM_SIZE << std::endl;
        std::cout << "============================================================================" << std::endl;
        std::cout << std::endl;

        return true;
    }
}


}

