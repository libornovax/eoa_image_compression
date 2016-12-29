//
// Libor Novak
// 10/11/2016
//

#ifndef ISHAPE_H
#define ISHAPE_H

#include <string>
#include <assert.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include "components/IVisitor.h"


namespace eic {


/**
 * @brief The SizeGroup enum
 * Each shape will belong into one of the groups, which will limit its mutation and crossover excess
 */
enum class SizeGroup {
    SMALL   = 0,
    MEDIUM  = 1,
    LARGE   = 2
};


/**
 * @brief The IShape class
 * Base class for all shape classes, wich can compose the final image
 */
class IShape
{
    friend class Mutator;
public:

    IShape (int r, int g, int b, int a, SizeGroup size_group);

    virtual std::shared_ptr<IShape> clone () const = 0;


    /**
     * @brief Accept method from the visitor design pattern
     */
    virtual void accept (IVisitor &visitor) = 0;

    /**
     * @brief Writes all parameters to a string
     * @return A string representation of the shape (solely for debug purposes)
     */
    virtual std::string print () const;

    /**
     * @brief Checks if the given circle is inside of the shape
     * @param center
     * @param radius
     * @return true if it is inside, false for outside
     */
    virtual bool contains (const cv::Point &center, int radius) const = 0;

    virtual bool intersects (const cv::Point &center, int radius) const = 0;

    virtual cv::Point getCenter () const = 0;

    virtual int getR () const final;
    virtual int getG () const final;
    virtual int getB () const final;
    virtual int getA () const final;
    virtual SizeGroup getSizeGroup () const final;

#ifdef USE_GPU
    /**
     * @brief Writes a float description of the shape into the given array. Max length 10!
     */
    virtual void writeDescription (float *desc_array) const = 0;
#endif


protected:

    /**
     * @brief Checks if all members contain feasible values
     */
    void _check () const;


    // -------------------------------------  PROTECTED MEMBERS  ------------------------------------- //
    int _r; // red
    int _g; // green
    int _b; // blue
    int _a; // alpha [0,100] %
    // Each shape will belong to a size group, which will limits its dimensions
    SizeGroup _size_group;

};


}


#endif // ISHAPE_H
