#include "IShape.h"

#include "components/RGen.h"
#include "components/utils.h"


namespace eic {


IShape::IShape (int r, int g, int b, int a)
    : _r(r),
      _g(g),
      _b(b),
      _a(a)
{
    this->_check();
}


void IShape::mutate ()
{
    std::normal_distribution<double> dist (0, 10); // mean, stddev

    this->_r += dist(RGen::mt());
    this->_g += dist(RGen::mt());
    this->_b += dist(RGen::mt());
    this->_a += dist(RGen::mt());

    // Correct the values to be in the interval [0,255]
    this->_r = utils::clip(this->_r, 0, 255);
    this->_g = utils::clip(this->_g, 0, 255);
    this->_b = utils::clip(this->_b, 0, 255);
    // Correct the value to be in the interval [0,100]
    this->_a = utils::clip(this->_a, 0, 100);
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

