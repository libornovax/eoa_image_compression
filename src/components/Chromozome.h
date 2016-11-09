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
#include "target.h"


namespace eic {


class Chromozome
{
public:

    Chromozome (const std::shared_ptr<const Target> &target);

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
    static std::shared_ptr<Chromozome> randomChromozome (const std::shared_ptr<const Target> &target);


    /**
     * @brief Length of the chromozome (number of shapes)
     */
    size_t size () const;

    /**
     * @brief Adds a new random shape to the chromozome
     * @param image_size Size of an image that is being approximated
     */
    void addRandomShape ();

    /**
     * @brief Returns pointer to one shape in the chromozome
     */
    std::shared_ptr<IShape>& operator[] (size_t i);
    const std::shared_ptr<IShape>& operator[] (size_t i) const;

    /**
     * @brief Computes the difference of the image represented by the chromozome and the target image
     * @return Difference, the lower, the better
     */
    double getFitness ();

    /**
     * @brief Triggers fitness recomputation in the next getFitness() call
     */
    void setDirty ();

    const std::shared_ptr<const Target>& getTarget () const;

    /**
     * @brief Renders the chromozome
     */
    cv::Mat asImage ();

    /**
     * @brief Accept method from the visitor design pattern
     */
    void accept (IVisitor &visitor);


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    std::vector<std::shared_ptr<IShape>> _chromozome;
    // Last computed difference from the target image
    double _fitness;
    // Whether the chromozome was touched and fitness needs recomputation
    bool _dirty;
    // Target image that we want to represent
    const std::shared_ptr<const Target> _target;

};


}


#endif // CHROMOZOME_H
