//
// Libor Novak
// 10/13/2016
//

#ifndef MUTATOR_H
#define MUTATOR_H

#include <opencv2/core/core.hpp>
#include "components/IVisitor.h"
#include "shapes/IShape.h"


namespace eic {


class Mutator : public IVisitor
{
public:

    Mutator (const cv::Size &image_size);


    /**
     * @brief Triggers mutation of the whole chromozome
     */
    virtual void visit (Chromozome &chromozome) override final;

    /**
     * @brief Mutates a circle shape
     */
    virtual void visit (Circle &circle) override final;

    /**
     * @brief Mutates a rectangle shape
     */
    virtual void visit (Rectangle &rect) override final;


private:

    /**
     * @brief Mutates members common to all shapes - colors
     * @param shape
     * @param mutated_feature Id of the feature to be mutated
     */
    void _mutateIShape (IShape &shape, int mutated_feature) const;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    cv::Size _image_size;

};


}


#endif // MUTATOR_H
