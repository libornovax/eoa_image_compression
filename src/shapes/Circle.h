//
// Libor Novak
// 10/11/2016
//

#ifndef CIRCLE_H
#define CIRCLE_H

#include <opencv2/core/core.hpp>
#include "IShape.h"


namespace eic {


class Circle : public IShape
{
public:

    Circle (int r, int g, int b/*, int a*/, int radius, const cv::Point2i &center);


    virtual void accept (IVisitor &visitor) const override final;

private:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    int _radius;
    cv::Point2i _center;

};


}


#endif // CIRCLE_H
