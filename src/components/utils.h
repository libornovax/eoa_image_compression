//
// Libor Novak
// 10/12/2016
//

#ifndef UTILS_H
#define UTILS_H

#include "RGen.h"


namespace eic {
namespace utils {


template<typename T>
/**
 * @brief Clip the given value to fit in the interval [min,max]
 * @param n Number to be clipped
 * @param min Lower bound
 * @param max Upper bound
 * @return Clipped number
 */
T clip (const T &n, const T &min, const T &max)
{
    return std::min(std::max(n, min), max);
}


bool makeMutation (double p)
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(RGen::mt()) <= p;
}


}
}


#endif // UTILS_H

