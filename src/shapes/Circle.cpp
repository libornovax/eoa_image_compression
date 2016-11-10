#include "Circle.h"

#include <iostream>
#include "components/RGen.h"


namespace eic {


Circle::Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center, CircleType type)
    : IShape(r, g, b, a),
      _radius(radius),
      _center(center),
      _type(type)
{
    this->_check();
}


std::shared_ptr<IShape> Circle::clone () const
{
    return std::make_shared<Circle>(this->_r, this->_g, this->_b, this->_a, this->_radius, this->_center, this->_type);
}


std::shared_ptr<Circle> Circle::randomCircle (const std::shared_ptr<const Target> &target)
{
    std::uniform_int_distribution<int> dista(30, 60);
    int a = dista(RGen::mt());

    std::uniform_int_distribution<int> distt(0, 2);
    CircleType t = CircleType(distt(RGen::mt()));

    auto minmax = Circle::radiusBounds(target->image_size, t);
    std::uniform_int_distribution<int> distr(minmax.first, minmax.second);
    int radius = distr(RGen::mt());

    std::uniform_int_distribution<int> distcx(0, target->image_size.width);
    std::uniform_int_distribution<int> distcy(0, target->image_size.height);
    cv::Point center(distcx(RGen::mt()), distcy(RGen::mt()));

    cv::Scalar rgb = Circle::extractColor(center, radius, target);

    return std::make_shared<Circle>(rgb[0], rgb[1], rgb[2], a, radius, center, t);
}


void Circle::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


std::string Circle::print () const
{
    std::string output = "CIRCLE { ";
    output += IShape::print() + ", ";
    output += "radius: " + std::to_string(this->_radius) + ", ";
    output += "center: [" + std::to_string(this->_center.x) + ", " + std::to_string(this->_center.y) + "], ";
    std::vector<std::string> names = { "SMALL", "MEDIUM", "LARGE" };
    output += "type: " + names[int(this->_type)];
    output += " }";
    return output;
}


bool Circle::contains (const cv::Point &p) const
{
    // Compute the distance of the point from the center and compare with radius
    double d = (this->_center.x-p.x)*(this->_center.x-p.x) + (this->_center.y-p.y)*(this->_center.y-p.y);

    return (d < this->_radius*this->_radius);
}


int Circle::getRadius () const
{
    return this->_radius;
}


const cv::Point& Circle::getCenter () const
{
    return this->_center;
}


CircleType Circle::getType () const
{
    return this->_type;
}


std::pair<int, int> Circle::radiusBounds (const cv::Size &image_size, CircleType t)
{
//    std::min(image_size.width, image_size.height)/4

    std::pair<int, int> minmax(0, 0);

    switch (t)
    {
    case CircleType::SMALL:
        minmax.first = 2;
        minmax.second = 10;
        break;
    case CircleType::MEDIUM:
        minmax.first = 8;
        minmax.second = 40;
        break;
    case CircleType::LARGE:
        minmax.first = 40;
        minmax.second = 300;
        break;
    default:
        std::cout << "Error: unknown circle type" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }

    return minmax;
}


cv::Scalar Circle::extractColor (const cv::Point2i &center, int radius,
                                 const std::shared_ptr<const Target> &target)
{
    return cv::Scalar(
            target->channels[0].at<uchar>(center.y, center.x),
            target->channels[1].at<uchar>(center.y, center.x),
            target->channels[2].at<uchar>(center.y, center.x));
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Circle::_check () const
{
    assert(this->_radius >= 0);

    IShape::_check();
}


}
