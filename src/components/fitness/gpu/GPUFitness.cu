#include "GPUFitness.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "components/Chromozome.h"


namespace eic {


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{

}


void computeFitnessGPU (const std::shared_ptr<Chromozome> &ch, bool write_channels)
{

}


}

