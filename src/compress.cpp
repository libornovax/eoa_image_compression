//
// Libor Novak
// 10/11/2016
//
// Converts an image to a vector representation, which is used to compress the image.
// This is code for EOA course at CTU in Prague.
//

#include <iostream>
#include <memory>

#include "shapes/Circle.h"
#include "components/Renderer.h"


int main (int argc, char* argv[])
{
    eic::Chromozome ch;
    ch.emplace_back(new eic::Circle(10, 10, 10, 50, cv::Point2i(0,0)));
    ch.emplace_back(new eic::Circle(0, 255, 128, 10, cv::Point2i(20,20)));
    ch.emplace_back(new eic::Circle(0, 0, 0, 15, cv::Point2i(40,40)));

    eic::Renderer r(cv::Size(40, 40));
    r.render(ch);


	std::cout << "hello" << std::endl;
	return 0;
}
