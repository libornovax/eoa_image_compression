//
// Libor Novak
// 12/29/2016
//

#ifndef CPUFITNESS_H
#define CPUFITNESS_H

#include <vector>


class Chromozome;

namespace eic {


void computeFitnessCPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels=false);

void computeFitnessCPU (const std::shared_ptr<Chromozome> &ch, bool write_channels=false);


}


#endif // CPUFITNESS_H
