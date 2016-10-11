//
// Libor Novak
// 10/11/2016
//

#ifndef ISHAPE_H
#define ISHAPE_H

#include <assert.h>
#include "components/IVisitor.h"


namespace eic {


class IShape
{
public:

    IShape (int r, int g, int b/*, int a*/);


    virtual void accept (IVisitor &visitor) const = 0;

protected:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PROTECTED MEMBERS  ------------------------------------- //
    int _r; // red
    int _g; // green
    int _b; // blue
//    int _a; // alpha

};


}


#endif // ISHAPE_H
