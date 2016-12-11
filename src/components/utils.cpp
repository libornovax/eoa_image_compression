#include "utils.h"


namespace eic {
namespace utils {


bool makeMutation (double p)
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(RGen::mt()) < p;
}


/**
 * @brief Finds indices of shapes in the chromozome, which intersect or contain the circle of interest
 * @param center, radius Parameters that define the circle of interest
 * @param chromozome
 * @return Vector of indices in the chromozome
 */
std::vector<int> findIntersectingShapesIdxs (const cv::Point &center, int radius,
                                             const std::shared_ptr<Chromozome> &chromozome)
{
    std::vector<int> intersecting_idxs;

    for (int i = 0; i < chromozome->size(); ++i)
    {
        // Add a small or medium shape if it intersects the circle
        if (chromozome->operator [](i)->getSizeGroup() != SizeGroup::LARGE &&
                chromozome->operator [](i)->intersects(center, radius))
        {
            intersecting_idxs.push_back(i);
        }
        // Add a large shape if it contains the whole circle
        if (chromozome->operator [](i)->getSizeGroup() == SizeGroup::LARGE &&
                chromozome->operator [](i)->contains(center, radius))
        {
            intersecting_idxs.push_back(i);
        }
    }

    return intersecting_idxs;
}


/**
 * @brief From the two chromozomes selects a random small or medium shape position as a random position
 * @param chromozome1
 * @param chromozome2
 * @return
 */
cv::Point selectRandomPositionForCrossover (const std::shared_ptr<Chromozome> &chromozome1,
                                            const std::shared_ptr<Chromozome> &chromozome2)
{
    // We do crossover in a way that we select a random small or medium shape, find all other shapes that
    // intersect it and then exchange those shapes. Here we select the random small or medium shape

    // Collect small (and medium) shapes from both chromozomes
    std::vector<std::shared_ptr<IShape>> small_shapes;
    for (int i = 0; i < chromozome1->size() && i < chromozome2->size(); ++i)
    {
        if (chromozome1->operator [](i)->getSizeGroup() != SizeGroup::LARGE)
        {
            small_shapes.push_back(chromozome1->operator [](i));
        }
        if (chromozome2->operator [](i)->getSizeGroup() != SizeGroup::LARGE)
        {
            small_shapes.push_back(chromozome2->operator [](i));
        }
    }

    std::uniform_int_distribution<int> dist(0, small_shapes.size()-1);

    return small_shapes[dist(RGen::mt())]->getCenter();
}


}
}
