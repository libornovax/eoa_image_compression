//
// Libor Novak
// 10/11/2016
//
// Converts an image to a vector representation, which is used to compress the image.
// This is code for EOA course at CTU in Prague.
//

#define _GLIBCXX_USE_CXX11_ABI 0

#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "shapes/Circle.h"
#include "components/Renderer.h"


int main (int argc, char* argv[])
{
    cv::Size image_size(400, 400);
    eic::Chromozome ch;
    for (int i = 0; i < 100; ++i)
    {
        ch.push_back(eic::Circle::randomCircle(image_size));
    }

    for (auto shape: ch)
    {
        std::cout << shape->print() << std::endl;
    }

    eic::Renderer r(image_size);

    auto start = std::chrono::system_clock::now();
    const std::vector<cv::Mat> channels = r.render(ch);
    auto end = std::chrono::system_clock::now();

    std::cout << "Elapsed: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) << " milliseconds" << std::endl;

    cv::Mat image;
    cv::merge(channels, image);

    cv::imshow("image", image);
    cv::waitKey();


	return 0;
}
