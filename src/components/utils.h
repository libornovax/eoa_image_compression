//
// Libor Novak
// 10/12/2016
//

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core/core.hpp>
#include "RGen.h"
#include "Chromozome.h"


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


/**
 * @brief Randomly returns true with the given probability p
 * @param p Probability of returning true
 * @return true/false
 */
bool makeMutation (double p);


/**
 * @brief Finds indices of shapes in the chromozome, which intersect or contain the circle of interest
 * @param center, radius Parameters that define the circle of interest
 * @param chromozome
 * @return Vector of indices in the chromozome
 */
std::vector<int> findIntersectingShapesIdxs (const cv::Point &center, int radius,
                                             const std::shared_ptr<Chromozome> &chromozome);


/**
 * @brief From the two chromozomes selects a random small or medium shape position as a random position
 * @param chromozome1
 * @param chromozome2
 * @return cv::Point coordinates
 */
cv::Point selectRandomPositionForCrossover (const std::shared_ptr<Chromozome> &chromozome1,
                                            const std::shared_ptr<Chromozome> &chromozome2);

}
}


#endif // UTILS_H

