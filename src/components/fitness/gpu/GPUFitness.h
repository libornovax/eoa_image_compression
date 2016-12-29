//
// Libor Novak
// 12/29/2016
//

#ifndef GPUFITNESS_H
#define GPUFITNESS_H

#include <vector>


class Chromozome;

namespace eic {


void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels=false);

void computeFitnessGPU (const std::shared_ptr<Chromozome> &ch, bool write_channels=false);


}


#endif // GPUFITNESS_H
