//
// Libor Novak
// 10/11/2016
//
// Converts an image to a vector representation, which is used to compress the image.
// This is code for EOA course at CTU in Prague.
//

#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "shapes/Circle.h"
#include "components/Renderer.h"
#include "algorithms/HillClimber.h"
#include "algorithms/DifferentialEvolution.h"
#include "algorithms/ClassicEA.h"
#include "components/Config.h"


void runCompression ()
{
    cv::Mat image = cv::imread(eic::Config::getParams().path_image, CV_LOAD_IMAGE_COLOR);
    cv::Size image_size = image.size();

    cv::imshow("original", image);
    cv::waitKey(1);

    cv::cvtColor(image, image, CV_BGR2RGB);
    std::vector<cv::Mat> image_channels;
    cv::split(image, image_channels);


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
    case eic::AlgorithmType::DIFFERENTIAL_EVOLUTION:
        {
            eic::DifferentialEvolution de(image_channels);
            result = de.run();
        }
        break;
    case eic::AlgorithmType::CLASSIC_EA:
        {
            eic::ClassicEA ea(image_channels);
            result = ea.run();
        }
        break;
    default:
        std::cout << "ERROR: Unsupported algorithm!" << std::endl;
        exit(EXIT_FAILURE);
        break;
    }


    // Save the resulting shapes to a file
    std::ofstream outfile(eic::Config::getParams().path_out + "/representation.txt");
    if (outfile)
    {
        for (int i = 0; i < result.size(); ++i)
        {
            outfile << result[i]->print() << std::endl;
        }
    }
    outfile.close();


    // Show the final approximated image
    cv::Mat approximation = result.asImage(image_size);

    cv::imwrite(eic::Config::getParams().path_out + "/approximation.png", approximation);
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
