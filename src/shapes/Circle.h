//
// Libor Novak
// 10/11/2016
//

#ifndef CIRCLE_H
#define CIRCLE_H

#include <memory>
#include <opencv2/core/core.hpp>
#include "IShape.h"
#include "components/target.h"


namespace eic {


class Circle : public IShape
{
    friend class Mutator;
public:

    Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center, SizeGroup size_group);

    virtual std::shared_ptr<IShape> clone () const override final;

    /**
     * @brief Generates a circle with random parameters (for initialization)
     * @param target Target image that we are composing
     * @return A new Circle instance
     */
    static std::shared_ptr<Circle> randomCircle (const std::shared_ptr<const Target> &target);


    virtual void accept (IVisitor &visitor) override final;

    virtual std::string print () const override final;

    virtual bool contains (const cv::Point &center, int radius) const override final;

    virtual bool intersects (const cv::Point &center, int radius) const override final;

    virtual int getRadius () const final;
    virtual cv::Point getCenter () const override final;

#ifdef USE_GPU
    virtual void writeDescription (int *desc_array) const override final;
#endif

    /**
     * @brief Min and max radius of a circle belonging to the given circle type
     * @param image_size
     * @param t
     * @return Pair of min, max
     */
    static std::pair<int, int> radiusBounds (const cv::Size &image_size, SizeGroup sg);

    /**
     * @brief Extracts color from the original images, which should be used for the specified circle
     * @param center Center of the circle
     * @param radius Radius of the circle
     * @param target Target image, which will be used for extracting colors
     * @return 3 element scalar representing RGB color
     */
    static cv::Scalar extractColor (const cv::Point2i &center, int radius,
                                    const std::shared_ptr<const Target> &target);

private:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    int _radius;
    cv::Point2i _center;

};


}


#endif // CIRCLE_H
