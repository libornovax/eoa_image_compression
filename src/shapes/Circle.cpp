#include "Circle.h"

#include "components/RGen.h"
#include "components/utils.h"


namespace eic {


Circle::Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center)
    : IShape(r, g, b, a),
      _radius(radius),
      _center(center)
{
    this->_check();
}


std::shared_ptr<Circle> Circle::randomCircle (const cv::Size &image_size)
{
    std::uniform_int_distribution<int> dist(0, 255);
    int r = dist(RGen::mt());
    int g = dist(RGen::mt());
    int b = dist(RGen::mt());

    std::uniform_int_distribution<int> dista(0, 100);
    int a = dista(RGen::mt());

    std::uniform_int_distribution<int> distr(1, std::max(image_size.width, image_size.height)/2);
    int radius = distr(RGen::mt());

    std::uniform_int_distribution<int> distcx(0, image_size.width);
    std::uniform_int_distribution<int> distcy(0, image_size.height);
    int x = distcx(RGen::mt());
    int y = distcy(RGen::mt());

    return std::make_shared<Circle>(r, g, b, a, radius, cv::Point(x, y));
}


void Circle::mutate ()
{
    std::normal_distribution<double> distr (0, 5); // mean, stddev
    this->_radius += distr(RGen::mt());
    this->_radius = utils::clip(this->_radius, 0, 10000); // Must be positive

    std::normal_distribution<double> distc (0, 10); // mean, stddev
    this->_center.x += distc(RGen::mt());
    this->_center.y += distc(RGen::mt());

    IShape::mutate();
}


void Circle::accept (IVisitor &visitor) const
{
    visitor.visit(*this);
}


std::string Circle::print () const
{
    std::string output = "CIRCLE { ";
    output += IShape::print() + ", ";
    output += "radius: " + std::to_string(this->_radius) + ", ";
    output += "center: [" + std::to_string(this->_center.x) + ", " + std::to_string(this->_center.y) + "]";
    output += " }";
    return output;
}


int Circle::getRadius () const
{
    return this->_radius;
}


const cv::Point& Circle::getCenter () const
{
    return this->_center;
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Circle::_check () const
{
    assert(this->_radius >= 0);

    IShape::_check();
}


}
