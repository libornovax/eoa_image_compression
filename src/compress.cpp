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


void runCompression ()
{
    cv::Mat image = cv::imread(eic::Config::getParams().path_image, CV_LOAD_IMAGE_COLOR);
    cv::Size image_size = image.size();

//    cv::cvtColor(image, image, CV_BGR2RGB); // This is how it should really be
    std::vector<cv::Mat> image_channels;
    cv::split(image, image_channels);


    cv::imshow("original", image);
    cv::waitKey(1);


    // Compress the image
    eic::Chromozome result;
    switch (eic::Config::getParams().algorithm)
    {
    case eic::AlgorithmType::HILL_CLIMBER:
        {
            eic::HillClimber hc(image_channels);
            result = hc.run();
        }
        break;
    default:
        std::cout << "ERROR: Unsupported algorithm!" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }


    // Show the final approximated image
    // Render it
    eic::Renderer r(image_size);
    const std::vector<cv::Mat> channels = r.render(result);

    // Merge the channels to one image
    cv::Mat approximation;
    cv::merge(channels, approximation);
//    cv::cvtColor(approximation, approximation, CV_RGB2BGR); // This is how it should really be

    cv::imwrite(eic::Config::getParams().path_out + "/approximation.jpg", approximation);
    cv::imshow("approximation", approximation);
    cv::waitKey();
}


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
    eic::Config::print();


    runCompression();


    return EXIT_SUCCESS;
}
