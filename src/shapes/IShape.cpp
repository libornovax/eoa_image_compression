#include "IShape.h"


namespace eic {


IShape::IShape (int r, int g, int b/*, int a*/)
    : _r(r),
      _g(g),
      _b(b)
{
    this->_check();
}


void IShape::_check () const
{
    assert(this->_r >= 0);
    assert(this->_r <= 255);
    assert(this->_g >= 0);
    assert(this->_g <= 255);
    assert(this->_b >= 0);
    assert(this->_b <= 255);
}


}

