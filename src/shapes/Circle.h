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


class Circle : public IShape
{
    friend class Mutator;
    friend class DifferentialCrossover;
public:

    Circle (int r, int g, int b, int a, int radius, const cv::Point2i &center);

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
