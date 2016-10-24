//
// Libor Novak
// 10/11/2016
//

#ifndef ISHAPE_H
#define ISHAPE_H

#include <string>
#include <assert.h>
#include <memory>
#include "components/IVisitor.h"


namespace eic {


/**
 * @brief The IShape class
 * Base class for all shape classes, wich can compose the final image
 */
class IShape
{
    friend class Mutator;
public:

    IShape (int r, int g, int b, int a);

    virtual std::shared_ptr<IShape> clone () const = 0;


    /**
     * @brief Accept method from the visitor design pattern
     */
    virtual void accept (IVisitor &visitor) = 0;

    /**
     * @brief Writes all parameters to a string
     * @return A string representation of the shape (solely for debug purposes)
     */
    virtual std::string print () const;

    virtual int getR () const final;
    virtual int getG () const final;
    virtual int getB () const final;
    virtual int getA () const final;


protected:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PROTECTED MEMBERS  ------------------------------------- //
    int _r; // red
    int _g; // green
    int _b; // blue
    int _a; // alpha [0,100] %

};


}


#endif // ISHAPE_H
