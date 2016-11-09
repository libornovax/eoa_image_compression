//
// Libor Novak
// 10/12/2016
//

#ifndef CHROMOZOME_H
#define CHROMOZOME_H

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include "shapes/IShape.h"


namespace eic {


class Chromozome
{
public:

    Chromozome ();

    /**
     * @brief Makes a deep copy of the chromozome and all its shapes
     * @return Shared pointer to a new Chromozome instance
     */
    std::shared_ptr<Chromozome> clone () const;

    /**
     * @brief Generates a random chromozome according to settings
     * @param image_size Size of an image that is being approximated
     * @return Shared pointer to a new Chromozome instance
     */
    static std::shared_ptr<Chromozome> randomChromozome (const cv::Size &image_size);


    /**
     * @brief Length of the chromozome (number of shapes)
     */
    size_t size () const;

    /**
     * @brief Adds a new random shape to the chromozome
     * @param image_size Size of an image that is being approximated
     */
    void addRandomShape (const cv::Size &image_size);

    /**
     * @brief Returns pointer to one shape in the chromozome
     */
    std::shared_ptr<IShape>& operator[] (size_t i);
    const std::shared_ptr<IShape>& operator[] (size_t i) const;

    /**
     * @brief Computes the difference of the image represented by the chromozome and the target image
     * @return Difference, the lower, the better
     */
    double computeDifference (const std::vector<cv::Mat> &target);

    /**
     * @brief Returns the latest computed difference
     */
    double getDifference () const;

    cv::Mat asImage (const cv::Size &image_size);

    /**
     * @brief Accept method from the visitor design pattern
     */
    void accept (IVisitor &visitor);


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    std::vector<std::shared_ptr<IShape>> _chromozome;
    // Last computed difference from the target image
    double _difference;

};


}


#endif // CHROMOZOME_H
