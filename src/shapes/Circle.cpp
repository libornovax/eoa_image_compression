#include "Circle.h"


namespace eic {


Circle::Circle (int r, int g, int b/*, int a*/, int radius, const cv::Point2i &center)
    : IShape(r, g, b/*, a*/),
      _radius(radius),
      _center(center)
{
    this->_check();
}


void Circle::accept (IVisitor &visitor) const
{
    visitor.visit(*this);
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Circle::_check () const
{
    assert(this->_radius >= 0);

    IShape::_check();
}


}
