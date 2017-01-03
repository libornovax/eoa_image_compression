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
// Limit for CUDA capability >= 2.0 is 48kB
// One pixel: 3*sizeof(int) = 12B, Dimensions of rendering window 60x60 = 3600, 3600*12 = 43200
#define CANVAS_DIMENSION 60
#define SHARED_MEM_SIZE 43200



#endif // SETTINGS_H

