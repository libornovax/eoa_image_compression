//
// Libor Novak
// 12/22/2016
//

#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <memory>
#include <opencv2/core/core.hpp>
#include "IShape.h"
#include "components/target.h"


namespace eic {


class Rectangle : public IShape
{
    friend class Mutator;
public:

    Rectangle (int r, int g, int b, int a, const cv::Rect &rect, SizeGroup size_group);

    virtual std::shared_ptr<IShape> clone () const override final;

    /**
     * @brief Generates a rectangle with random parameters (for initialization)
     * @param target Target image that we are composing
     * @return A new Rectangle instance
     */
    static std::shared_ptr<Rectangle> randomRectangle (const std::shared_ptr<const Target> &target);


    virtual void accept (IVisitor &visitor) override final;

    virtual std::string print () const override final;

    virtual bool contains (const cv::Point &center, int radius) const override final;

    virtual bool intersects (const cv::Point &center, int radius) const override final;

    virtual const cv::Rect& getRect () const final;
    virtual cv::Point getCenter () const override final;

    /**
     * @brief Min and max width (height) of a rectangle belonging to the given rectangle size group
     * @param image_size
     * @param sg Size group
     * @return Pair of min, max
     */
    static std::pair<int, int> whBounds (const cv::Size &image_size, SizeGroup sg);

    /**
     * @brief Extracts color, which should be used for the specified rectangle, from the original image
     * @param rect
     * @param target Target image, which will be used for extracting colors
     * @return 3 element scalar representing RGB color
     */
    static cv::Scalar extractColor (const cv::Rect &rect, const std::shared_ptr<const Target> &target);

private:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    cv::Rect _rect;

};


}


#endif // RECTANGLE_H
