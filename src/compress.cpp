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
    cv::Size image_size(40, 40);
    eic::Chromozome ch;
    ch.push_back(eic::Circle::randomCircle(image_size));
    ch.push_back(eic::Circle::randomCircle(image_size));
    ch.push_back(eic::Circle::randomCircle(image_size));

    for (auto shape: ch)
    {
        std::cout << shape->print() << std::endl;
    }

    eic::Renderer r(cv::Size(40, 40));
    r.render(ch);


	std::cout << "hello" << std::endl;
	return 0;
}
