//
// Libor Novak
// 10/11/2016
//

#ifndef CIRCLE_H
#define CIRCLE_H

#include <memory>
#include <opencv2/core/core.hpp>
#include "IShape.h"


namespace eic {


enum class CircleType {
    SMALL   = 0,
    MEDIUM  = 1,
    LARGE   = 2
};


class Circle : public IShape
{
    friend class Mutator;
    friend class DifferentialCrossover;
public:

    Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center, CircleType type);

    virtual std::shared_ptr<IShape> clone () const override final;

    /**
     * @brief Generates a circle with random parameters (for initialization)
     * @param image_size Size of the image we are composing
     * @return A new Circle instance
     */
    static std::shared_ptr<Circle> randomCircle (const cv::Size &image_size);


    virtual void accept (IVisitor &visitor) override final;

    virtual std::string print () const override final;

    virtual bool contains (const cv::Point &p) const override final;

    virtual int getRadius () const final;
    virtual const cv::Point& getCenter () const final;
    virtual CircleType getType () const final;

    /**
     * @brief Min and max radius of a circle belonging to the given circle type
     * @param image_size
     * @param t
     * @return Pair of min, max
     */
    static std::pair<int, int> radiusBounds (const cv::Size &image_size, CircleType t);

private:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    int _radius;
    cv::Point2i _center;
    CircleType _type;

};


}


#endif // CIRCLE_H
