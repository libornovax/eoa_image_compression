//
// Libor Novak
// 11/08/2016
//

#ifndef ONEPOINTCROSSOVER_H
#define ONEPOINTCROSSOVER_H

#include <opencv2/core/core.hpp>
#include "components/IVisitor.h"
#include "shapes/IShape.h"


namespace eic {


/**
 * @brief The OnePointCrossover class
 * The "One Point" is a little misleading in the name. The crossover works as follows: We select a random
 * point in the image (random coordinates) and select all shapes that intersect the given point. Then we
 * exchange those shapes between the chromozomes.
 */
class OnePointCrossover : public IVisitor
{
public:

    OnePointCrossover (const std::shared_ptr<Chromozome> &x);


    /**
     * @brief Triggers crossover with the chromozomes
     */
    virtual void visit (Chromozome &chromozome) override final;

    /**
     * @brief Crossover of a circle shape
     */
    virtual void visit (Circle &circle) override final;


private:

    /**
     * @brief Finds indices of shapes in the chromozome, which contain the given point
     * @param p Point
     * @param chromozome
     * @return Vector of indices in the chromozome
     */
    static std::vector<int> _containingIdxs (const cv::Point &p, const Chromozome &chromozome);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Chomozome, which will be used for the crossover
    const std::shared_ptr<Chromozome> _x;

};


}


#endif // ONEPOINTCROSSOVER_H
