//
// Libor Novak
// 11/07/2016
//

#ifndef DIFFERENTIALCROSSOVER_H
#define DIFFERENTIALCROSSOVER_H

#include <opencv2/core/core.hpp>
#include "components/IVisitor.h"
#include "shapes/IShape.h"


namespace eic {


class DifferentialCrossover : public IVisitor
{
public:

    DifferentialCrossover (const cv::Size &image_size, const std::shared_ptr<Chromozome> &x1,
                           const std::shared_ptr<Chromozome> &x2, const std::shared_ptr<Chromozome> &x3);


    /**
     * @brief Triggers crossover with the chromozomes
     */
    virtual void visit (Chromozome &chromozome) override final;

    /**
     * @brief Differerntial crossover of a circle shape
     */
    virtual void visit (Circle &circle) override final;


private:

    static const std::shared_ptr<Circle> _findClosestCircle (const Circle &circle,
                                                             const std::shared_ptr<Chromozome> ch);


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    cv::Size _image_size;
    // Differential crossover combines 3 different individuals and implants the combination into the original
    const std::shared_ptr<Chromozome> _x1;
    const std::shared_ptr<Chromozome> _x2;
    const std::shared_ptr<Chromozome> _x3;

};


}


#endif // DIFFERENTIALCROSSOVER_H
