#include "check_error.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


void handle_error(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        std::cout << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

