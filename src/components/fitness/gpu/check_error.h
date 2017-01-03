#ifndef CHECK_ERROR_H
#define CHECK_ERROR_H


void handle_error(cudaError_t error, const char *file, int line);

// Macro for checking CUDA errors
#define CHECK_ERROR(error) (handle_error(error, __FILE__, __LINE__))


#endif // CHECK_ERROR_H

