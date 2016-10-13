#ifndef MUTATOR_H
#define MUTATOR_H

#include "components/IVisitor.h"
#include "shapes/IShape.h"


namespace eic {


class Mutator : public IVisitor
{
public:

    Mutator ();


    /**
     * @brief Triggers mutation of the whole chromozome
     */
    virtual void visit (Chromozome &chromozome) override final;

    /**
     * @brief Mutates a circle shape
     */
    virtual void visit (Circle &circle) override final;


private:

    /**
     * @brief Mutates members common to all shapes - colors
     * @param shape
     */
    void _mutateIShape (IShape &shape) const;

};


}


#endif // MUTATOR_H
