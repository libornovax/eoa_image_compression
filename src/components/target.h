#ifndef TARGET_H
#define TARGET_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace eic {


/**
 * @brief The Target struct
 * Struct for storing the target image
 */
struct Target {

    Target (const cv::Mat &bgr_image_in, const cv::Mat &grayscale_weights, double max_weight)
        : bgr_image(bgr_image_in)
    {
        image_size = bgr_image.size();

        // Convert to RGB
        cv::cvtColor(bgr_image, image, CV_BGR2RGB);
        // Extract full detail channels
        cv::split(image, channels);

        // Blur the image - BLURRED channels
        cv::GaussianBlur(image, blurred_image, cv::Size(5, 5), 2);
        cv::split(blurred_image, blurred_channels);


        // -- WEIGHTS -- //
        // Weight map of the image is a combination of the provided weight file and detected edges.
        // The detected edges have a max of 1/10 of the max of the provided weight map

        // Weight map from file
        cv::Mat external_weights;
        {
            grayscale_weights.convertTo(external_weights, CV_32FC1);
            double max_val, dummy; cv::minMaxLoc(external_weights, &dummy, &max_val);
            external_weights *= max_weight/max_val;
            external_weights += 1;
        }
        // Edges
        cv::Mat edge_weights;
        {
            cv::Canny(blurred_image, edge_weights, 600, 1000, 5, true);
            edge_weights.convertTo(edge_weights, CV_32FC1);
            cv::GaussianBlur(edge_weights, edge_weights, cv::Size(19, 19), 5);
            double max_val, dummy; cv::minMaxLoc(edge_weights, &dummy, &max_val);
            edge_weights *= max_weight/max_val/10;  // 1/10 of the max_weight
            edge_weights += 1;
        }

        weights = cv::max(external_weights, edge_weights);

//        {
//            cv::imshow("original", bgr_image);
//            cv::imshow("weights", (weights-1)*(1.0/max_weight));
//            cv::imshow("blurred", blurred_image);
//            cv::waitKey(1);
//        }
    }


    // ------------------------------------------  DATA MEMBERS  ------------------------------------------ //
    cv::Size                image_size;
    // Full detail image
    const cv::Mat           bgr_image;
    cv::Mat                 image; // RGB
    std::vector<cv::Mat>    channels;
    // Blurred image
    cv::Mat                 blurred_image;
    std::vector<cv::Mat>    blurred_channels;
    cv::Mat                 weights;
};


}


#endif // TARGET_H

