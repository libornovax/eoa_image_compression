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
#include "components/fitness/cpu/CPUFitness.h"
#ifdef USE_GPU
#include "components/fitness/gpu/GPUFitness.h"
#endif


namespace eic {


class Chromozome
{
    friend class Mutator;
    friend void computeFitnessCPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels);
    friend void computeFitnessCPU (const std::shared_ptr<Chromozome> &ch, bool write_channels);
#ifdef USE_GPU
    friend void computeFitnessGPU (const std::vector<std::shared_ptr<Chromozome>> &chromozomes, bool write_channels);
    friend void computeFitnessGPU (const std::shared_ptr<Chromozome> &ch, bool write_channels);
#endif
public:

    Chromozome (const std::shared_ptr<const Target> &target, const cv::Rect roi);

    /**
     * @brief Makes a deep copy of the chromozome and all its shapes
     * @return Shared pointer to a new Chromozome instance
     */
    std::shared_ptr<Chromozome> clone () const;

    /**
     * @brief Copies the other chromozome into itself (updates itself with the data from the other chromozome)
     * @param other Chromozome
     */
    void update (const std::shared_ptr<Chromozome> &other);

    /**
     * @brief Generates a random chromozome according to settings
     * @param image_size Size of an image that is being approximated
     * @return Shared pointer to a new Chromozome instance
     */
    static std::shared_ptr<Chromozome> randomChromozome (const std::shared_ptr<const Target> &target);


    /**
     * @brief Sorts the shapes in the chromozome by size
     */
    void sort ();

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
     * @brief Activates the computation of error on the roi
     */
    void activateROI ();
    void deactivateROI ();

    /**
     * @brief Renders the chromozome
     */
    cv::Mat asImage ();

    std::vector<cv::Mat>& channels ();

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
    // Rendered channels
    std::vector<cv::Mat> _channels;
    // Target image that we want to represent
    std::shared_ptr<const Target> _target;
    // Region of interest in the target image that this chromozome specializes on
    cv::Rect _roi;
    // Whether we should consider extra error from the roi or not
    bool _roi_active;

};


}


#endif // CHROMOZOME_H
