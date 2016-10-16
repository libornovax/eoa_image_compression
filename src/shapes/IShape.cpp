#include "IShape.h"


namespace eic {


IShape::IShape (int r, int g, int b, int a)
    : _r(r),
      _g(g),
      _b(b),
      _a(a),
      _is_new(true)
{
    this->_check();
}


std::string IShape::print () const
{
    std::string output = "rgba: (";
    output += std::to_string(this->_r) + ", ";
    output += std::to_string(this->_g) + ", ";
    output += std::to_string(this->_b) + ", ";
    output += std::to_string(this->_a) + ")";
    return output;
}


bool IShape::isNew () const
{
    return this->_is_new;
}


void IShape::setOld ()
{
    this->_is_new = false;
}


int IShape::getR () const
{
    return this->_r;
}


int IShape::getG () const
{
    return this->_g;
}


int IShape::getB () const
{
    return this->_b;
}


int IShape::getA () const
{
    return this->_a;
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void IShape::_check () const
{
    assert(this->_r >= 0);
    assert(this->_r <= 255);
    assert(this->_g >= 0);
    assert(this->_g <= 255);
    assert(this->_b >= 0);
    assert(this->_b <= 255);
    assert(this->_a >= 0);
    assert(this->_a <= 100);
}


}

