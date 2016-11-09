#ifndef TARGET_H
#define TARGET_H

#include <opencv2/core/core.hpp>


namespace eic {


/**
 * @brief The Target struct
 * Struct for storing the target image
 */
struct Target {
    std::vector<cv::Mat> channels;
    cv::Size image_size;
};


}


#endif // TARGET_H

