#include "GPUFitness.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "components/Chromozome.h"
#include "check_error.h"
#include "kernels.h"
#include "settings.h"


namespace eic {


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{
    uchar* g_target; cudaMalloc((void**)&g_target, 10*sizeof(uchar));
//    cudaMemcpy(g_seq_in_out, seq.data(), seq.size()*sizeof(float), cudaMemcpyHostToDevice);

    float* g_out_fitness; cudaMalloc((void**)&g_out_fitness, 10*sizeof(float));

    int population[] = { 1/*roi*/, 1, 255, 120, 160, 30, 600, 350, 200 };
    int* g_population; cudaMalloc((void**)&g_population, 11*sizeof(int));
    cudaMemcpy(g_population, population, 11*sizeof(int), cudaMemcpyHostToDevice);

    int* g_canvas; cudaMalloc((void**)&g_canvas, 1000*600*3*sizeof(int));


    // Each rendering can run only on one multiprocessor!!! Because of the shared memory
    populationFitness<<< 1, 64, SHARED_MEM_SIZE >>>(g_target, 1000, 600, g_population, 1, 1, g_out_fitness, g_canvas);

    cv::Mat canvas(600, 1000, CV_32SC3);
    cudaMemcpy(canvas.ptr<int>(), g_canvas, 1000*600*3*sizeof(int), cudaMemcpyDeviceToHost);

//    std::cout << canvas << std::endl;

    canvas.convertTo(canvas, CV_8UC3);
    cv::cvtColor(canvas, canvas, CV_RGB2BGR);
    cv::imwrite("render.png", canvas);

    cudaFree(g_target);
    cudaFree(g_out_fitness);
    cudaFree(g_population);
    cudaFree(g_canvas);

    exit(EXIT_SUCCESS);
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

