//
// Libor Novak
// 12/29/2016
//
// Settings of the GPU fitness computation
//

#ifndef SETTINGS_H
#define SETTINGS_H

// Max length of description of a shape, which is passed to the GPU
// A chromozome is passed to the GPU as a compact array of integers, this is the number of ints, which are
// taken by one shape
#define DESC_LEN 10


// SHARED MEMORY SIZE
// Shared memory will be split between canvas and chromozome representation
#ifdef RENDER_AVERAGE
    // Limit for CUDA capability >= 2.0 is 48kB
    // One pixel: 4*sizeof(int) = 16B, Dimensions of rendering window 46x46 = 2116, 2116*16 = 33856
    #define CANVAS_DIMENSION 46
    #define CANVAS_MEM_SIZE 33856
    // Chromozome of length 250 * DESC_LEN * sizeof(int) = 10000
    #define CHROMOZOME_MEM_SIZE 10000
#else
    // Limit for CUDA capability >= 2.0 is 48kB
    // One pixel: 3*sizeof(int) = 12B, Dimensions of rendering window 55x55 = 3025, 3025*12 = 36300
    #define CANVAS_DIMENSION 55
    #define CANVAS_MEM_SIZE 36300
    // Chromozome of length 250 * DESC_LEN * sizeof(int) = 10000
    #define CHROMOZOME_MEM_SIZE 10000
#endif


// Because we cannot determine the number of cuda cores from device properties, we need to set this by hand.
#define THREADS_PER_BLOCK 128


#endif // SETTINGS_H

