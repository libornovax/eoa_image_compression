#include "GPUFitness.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "components/Chromozome.h"
#include "check_error.h"
#include "kernels.h"


namespace eic {


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{
    uchar* g_target; cudaMalloc((void**)&g_target, 10*sizeof(uchar));
//    cudaMemcpy(g_seq_in_out, seq.data(), seq.size()*sizeof(float), cudaMemcpyHostToDevice);

    float* g_out_fitness; cudaMalloc((void**)&g_out_fitness, 10*sizeof(float));

    float population[] = { 1.0f/*roi*/, 1.0f, 255.0f, 120.0f, 160.0f, 30.0f, 25.0f, 35.0f, 10.0f };
    float* g_population; cudaMalloc((void**)&g_population, 11*sizeof(float));
    cudaMemcpy(g_population, population, 11*sizeof(float), cudaMemcpyHostToDevice);

    float* g_canvas; cudaMalloc((void**)&g_canvas, 80*50*3*sizeof(float));


    populationFitness<<< 1, 64, 80*50*3*sizeof(float) >>>(g_target, 80, 50, g_population, 1, 1, g_out_fitness, g_canvas);

    cv::Mat canvas(50, 80, CV_32FC3);
    cudaMemcpy(canvas.ptr<float>(), g_canvas, 80*50*3*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << canvas << std::endl;

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


}

