#include "Rectangle.h"

#include <iostream>
#include "components/RGen.h"


namespace eic {


Rectangle::Rectangle (int r, int g, int b, int a, const cv::Rect &rect, SizeGroup size_group)
    : IShape(r, g, b, a, size_group),
      _rect(rect)
{
    this->_check();
}


std::shared_ptr<IShape> Rectangle::clone () const
{
    return std::make_shared<Rectangle>(this->_r, this->_g, this->_b, this->_a, this->_rect, this->_size_group);
}


std::shared_ptr<Rectangle> Rectangle::randomRectangle (const std::shared_ptr<const Target> &target)
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

    auto minmax = Rectangle::whBounds(target->image_size, sg);
    std::uniform_int_distribution<int> distwh(minmax.first, minmax.second);
    int width  = distwh(RGen::mt());
    int height = distwh(RGen::mt());

    std::uniform_int_distribution<int> distx(0, target->image_size.width-width);
    std::uniform_int_distribution<int> disty(0, target->image_size.height-height);
    cv::Rect rect(distx(RGen::mt()), disty(RGen::mt()), width, height);

    cv::Scalar rgb = Rectangle::extractColor(rect, target);

    return std::make_shared<Rectangle>(rgb[0], rgb[1], rgb[2], a, rect, sg);
}


void Rectangle::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


std::string Rectangle::print () const
{
    std::string output = "RECTANGLE { ";
    output += IShape::print() + ", ";
    std::stringstream ss; ss << this->_rect;
    output += "rect: " + ss.str() + ", ";
    std::vector<std::string> names = { "SMALL", "MEDIUM", "LARGE" };
    output += "type: " + names[int(this->_size_group)];
    output += " }";
    return output;
}


bool Rectangle::contains (const cv::Point &center, int radius) const
{
    return (center.x-radius > this->_rect.x && center.y-radius > this->_rect.y &&
            center.x+radius < this->_rect.br().x && center.y+radius < this->_rect.br().y);
}


bool Rectangle::intersects (const cv::Point &center, int radius) const
{
    return (center.x+radius > this->_rect.x && center.y+radius > this->_rect.y &&
            center.x-radius < this->_rect.br().x && center.y-radius < this->_rect.br().y);
}


const cv::Rect& Rectangle::getRect () const
{
    return this->_rect;
}


cv::Point Rectangle::getCenter() const
{
    return (this->_rect.tl() + this->_rect.br()) * 0.5;
}


std::pair<int, int> Rectangle::whBounds (const cv::Size &image_size, SizeGroup sg)
{
    std::pair<int, int> minmax(0, 0);

    switch (sg)
    {
    case SizeGroup::SMALL:
        minmax.first = 4;
        minmax.second = 20;
        break;
    case SizeGroup::MEDIUM:
        minmax.first = 16;
        minmax.second = 80;
        break;
    case SizeGroup::LARGE:
        minmax.first = 80;
        minmax.second = std::min(image_size.width, image_size.height)/2;
        break;
    default:
        std::cout << "Error: unknown SizeGroup" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }

    return minmax;
}


cv::Scalar Rectangle::extractColor (const cv::Rect &rect, const std::shared_ptr<const Target> &target)
{
    cv::Point center = (rect.tl() + rect.br()) * 0.5;

    return cv::Scalar(
            target->channels[0].at<uchar>(center.y, center.x),
            target->channels[1].at<uchar>(center.y, center.x),
            target->channels[2].at<uchar>(center.y, center.x));
}



// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void Rectangle::_check () const
{
    assert(this->_rect.x >= 0);
    assert(this->_rect.y >= 0);
    assert(this->_rect.width > 0);
    assert(this->_rect.height > 0);

    IShape::_check();
}


}
