#include "CPUFitness.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Renderer.h"
#include "components/Chromozome.h"


namespace eic {

namespace {

    /**
     * @brief Computes weighted difference between the original image and the candidate
     * @param target Original image
     * @param render Candidate rendered from a chromozome
     * @param weights Weight map of the pixels
     * @return Weighted difference
     */
    double computeDifference (const cv::Mat &target, const cv::Mat &render, const cv::Mat &weights)
    {
        cv::Mat diff;
        cv::subtract(target, render, diff, cv::noArray(), CV_32FC1);
        cv::pow(diff, 2, diff);

        // Apply weight on each image pixel
        cv::multiply(diff, weights, diff);

        // Tolerate small differences in color - only count as error if the difference is above the set
        // threshold
        cv::threshold(diff, diff, 50, 0, CV_THRESH_TOZERO);

        cv::Scalar total = cv::sum(diff);
        return total[0];
    }

}


//template<typename CH>
void computeFitnessCPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels)
{
    for (const std::shared_ptr<Chromozome> &ch: chromozomes)
    {
        computeFitnessCPU(ch, write_channels);
    }
}


//template<typename CH>
void computeFitnessCPU (const std::shared_ptr<Chromozome> &ch, bool write_channels)
{
    // Render and compute fitness
    assert(ch->_target->channels.size() == 3);
    assert(ch->_target->channels[0].size() == ch->_target->channels[1].size() && ch->_target->channels[0].size() == ch->_target->channels[2].size());

    // Render the image
    Renderer renderer(ch->_target->image_size);
    std::vector<cv::Mat> channels = renderer.render(*ch);

    // Compute pixel-wise difference
    double fitness = 0;
    for (size_t i = 0; i < 3; ++i)
    {
        if (ch->_roi_active)
        {
            double roi_weight = double(ch->_target->image_size.area()) / ch->_roi.area();
            // ROI is activated, include error from the ROI as well
            fitness += roi_weight * computeDifference(ch->_target->blurred_channels[i](ch->_roi),
                                                      channels[i](ch->_roi),
                                                      ch->_target->weights(ch->_roi));
        }
        fitness += computeDifference(ch->_target->blurred_channels[i],
                                     channels[i], ch->_target->weights);
    }

    if (write_channels)
    {
        // Copy also the rendered channels
        ch->_channels = channels;
    }
    ch->_fitness = fitness;
    ch->_dirty = false;
}


}


