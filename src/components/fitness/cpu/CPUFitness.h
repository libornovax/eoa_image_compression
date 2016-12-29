//
// Libor Novak
// 12/29/2016
//

#ifndef CPUFITNESS_H
#define CPUFITNESS_H

#include <vector>


namespace eic {
namespace CPUFitness {

    template<typename CH>
    void computeFitness (const std::vector<std::shared_ptr<CH>> &chromozomes, bool write_channels=false);

    template<typename CH>
    void computeFitness (const std::shared_ptr<CH> &ch, bool write_channels=false);

}
}


#include "CPUFitness.cpp"

#endif // CPUFITNESS_H
