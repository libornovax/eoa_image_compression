//
// Libor Novak
// 10/11/2016
//
// Converts an image to a vector representation, which is used to compress the image.
// This is code for EOA course at CTU in Prague.
//

#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>

#include "shapes/Circle.h"
#include "components/Renderer.h"
#include "algorithms/HillClimber.h"
#include "components/Config.h"


int main (int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "ERROR: Missing config file!" << std::endl;
        std::cout << "Usage: ./compress path/to/some_config.yaml" << std::endl;
        return EXIT_FAILURE;
    }

    // Load the config file
    std::string path_config(argv[1]);
    eic::Config::loadParams(path_config);



    cv::Mat image = cv::imread("test.jpg", CV_LOAD_IMAGE_COLOR);
    cv::Size image_size = image.size();

//    cv::cvtColor(image, image, CV_BGR2RGB);
    std::vector<cv::Mat> image_channels;
    cv::split(image, image_channels);

    cv::imshow("original", image);
    cv::waitKey(1);


    eic::HillClimber hc(image_channels);
    eic::Chromozome result = hc.run();











//    eic::Chromozome ch;
//    for (int i = 0; i < 100; ++i)
//    {
//        ch.chromozome().push_back(eic::Circle::randomCircle(image_size));
//    }

////    for (auto shape: ch)
////    {
////        std::cout << shape->print() << std::endl;
////    }

    eic::Renderer r(image_size);

//    auto start = std::chrono::system_clock::now();
    const std::vector<cv::Mat> channels = r.render(result);
//    auto end = std::chrono::system_clock::now();

//    std::cout << "Elapsed: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()) << " milliseconds" << std::endl;
//    std::cout << "Difference: " << result.getDifference() << std::endl;

    cv::Mat approximation;
    cv::merge(channels, approximation);

    cv::imshow("approximation", approximation);
    cv::imwrite("approximation.jpg", approximation);
    cv::waitKey();


    return EXIT_SUCCESS;
}
