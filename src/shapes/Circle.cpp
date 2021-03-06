#include "Circle.h"

#include <iostream>
#include "components/RGen.h"
#include "components/Config.h"


namespace eic {


Circle::Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center, SizeGroup size_group)
    : IShape(r, g, b, a, size_group),
      _radius(radius),
      _center(center)
{
    this->_check();
}


std::shared_ptr<IShape> Circle::clone () const
{
    return std::make_shared<Circle>(this->_r, this->_g, this->_b, this->_a, this->_radius, this->_center, this->_size_group);
}


std::shared_ptr<Circle> Circle::randomCircle (const std::shared_ptr<const Target> &target)
{
#ifdef RENDER_AVERAGE
    // If rendering average, allow alpha to go higher - more pronounced color
    std::uniform_int_distribution<int> dista(30, 600);
#else
    std::uniform_int_distribution<int> dista(30, 60);
#endif
    int a = dista(RGen::mt());

    std::uniform_int_distribution<int> distt(0, 2);
    SizeGroup sg = SizeGroup(distt(RGen::mt()));

    auto minmax = Circle::radiusBounds(target->image_size, sg);
    std::uniform_int_distribution<int> distr(minmax.first, minmax.second);
    int radius = distr(RGen::mt());

    std::uniform_int_distribution<int> distcx(0, target->image_size.width);
    std::uniform_int_distribution<int> distcy(0, target->image_size.height);
    cv::Point center(distcx(RGen::mt()), distcy(RGen::mt()));

    cv::Scalar rgb = Circle::extractColor(center, radius, target);

    return std::make_shared<Circle>(rgb[0], rgb[1], rgb[2], a, radius, center, sg);
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
    output += "type: " + names[int(this->_size_group)];
    output += " }";
    return output;
}


bool Circle::contains (const cv::Point &center, int radius) const
{
    // Compute the distance of the point from the center and compare with radius
    double d = std::sqrt(double((this->_center.x-center.x)*(this->_center.x-center.x) + (this->_center.y-center.y)*(this->_center.y-center.y)));
    return (d+radius < this->_radius);
}


bool Circle::intersects (const cv::Point &center, int radius) const
{
    double d = (center.x-this->_center.x)*(center.x-this->_center.x) + (center.y-this->_center.y)*(center.y-this->_center.y);
    return (d < (this->_radius + radius)*(this->_radius + radius));
}


int Circle::getRadius () const
{
    return this->_radius;
}


cv::Point Circle::getCenter() const
{
    return this->_center;
}


#ifdef USE_GPU
void Circle::writeDescription (int *desc_array) const
{
    desc_array[0] = int(ShapeType::CIRCLE);
    IShape::writeDescription(desc_array);  // RGBa
    desc_array[5] = this->_center.x;
    desc_array[6] = this->_center.y;
    desc_array[7] = this->_radius;
}
#endif


std::pair<int, int> Circle::radiusBounds (const cv::Size &image_size, SizeGroup sg)
{
    std::pair<int, int> minmax(0, 0);

    int dim = std::min(image_size.width, image_size.height);

    switch (sg)
    {
    case SizeGroup::SMALL:
        minmax.first  = std::max(1, dim/200);
        minmax.second = std::max(2.0, ceil(dim/40.0));
        break;
    case SizeGroup::MEDIUM:
        minmax.first  = std::max(1, dim/50);
        minmax.second = std::max(2.0, ceil(dim/10.0));
        break;
    case SizeGroup::LARGE:
        minmax.first  = std::max(1, dim/10);
        minmax.second = std::max(2.0, ceil(dim/4.0));
        break;
    default:
        std::cout << "Error: unknown SizeGroup" << std::endl;
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
