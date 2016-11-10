#ifndef TARGET_H
#define TARGET_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// The edge weight map maximum
#define EDGE_WEIGHT 100


namespace eic {


/**
 * @brief The Target struct
 * Struct for storing the target image
 */
struct Target {

    Target (const cv::Mat &bgr_image_in)
        : bgr_image(bgr_image_in)
    {
        image_size = bgr_image.size();

        // Convert to RGB
        cv::cvtColor(bgr_image, image, CV_BGR2RGB);
        // Extract full detail channels
        cv::split(image, channels);

        // Blur the image - BLURRED channels
        cv::GaussianBlur(image, blurred_image, cv::Size(9, 9), 2);
        cv::split(blurred_image, blurred_channels);

        // Edges
        cv::Canny(blurred_image, weights, 600, 1000, 5, true);
        weights.convertTo(weights, CV_32FC1);
        cv::GaussianBlur(weights, weights, cv::Size(19, 19), 5);
        double max_val, dummy; cv::minMaxLoc(weights, &dummy, &max_val);
        weights *= EDGE_WEIGHT/max_val;
        weights += 1;

        {
            cv::imshow("original", bgr_image);
            cv::imshow("weights", weights-1);
            cv::imshow("blurred", blurred_image);
            cv::waitKey(1);
        }
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

