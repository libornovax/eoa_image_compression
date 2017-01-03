#include "IShape.h"


namespace eic {


IShape::IShape (int r, int g, int b, int a, SizeGroup size_group)
    : _r(r),
      _g(g),
      _b(b),
      _a(a),
      _size_group(size_group)
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


SizeGroup IShape::getSizeGroup () const
{
    return this->_size_group;
}


#ifdef USE_GPU
void IShape::writeDescription (float *desc_array) const
{
    desc_array[1] = float(this->_r);
    desc_array[2] = float(this->_g);
    desc_array[3] = float(this->_b);
    desc_array[4] = float(this->_a);
}
#endif


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
#ifdef RENDER_AVERAGE
    assert(this->_a <= 600);
#else
    assert(this->_a <= 100);
#endif
}


}

