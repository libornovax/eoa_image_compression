#include "Chromozome.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Renderer.h"
#include "Config.h"
#include "shapes/Circle.h"
#include "RGen.h"


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


Chromozome::Chromozome(const std::shared_ptr<const Target> &target, const cv::Rect roi)
    : _fitness(DBL_MAX),
      _dirty(true),
      _target(target),
      _age(0),
      _roi(roi)
{

}


std::shared_ptr<Chromozome> Chromozome::clone () const
{
    auto ch = std::make_shared<Chromozome>(this->_target, this->_roi);

    for (auto &shape: this->_chromozome)
    {
        ch->_chromozome.push_back(shape->clone());
    }

    ch->_fitness = this->_fitness;
    ch->_dirty   = this->_dirty;
    ch->_age     = this->_age;

    return ch;
}


std::shared_ptr<Chromozome> Chromozome::randomChromozome (const std::shared_ptr<const Target> &target)
{
    // Select a random region of interest in target
    int square_dim = std::min(target->image_size.width, target->image_size.height) / 2;
    std::uniform_int_distribution<int> distw(0, target->image_size.width-square_dim);
    std::uniform_int_distribution<int> disth(0, target->image_size.height-square_dim);
    cv::Rect roi(distw(RGen::mt()), disth(RGen::mt()), square_dim, square_dim);

    auto ch = std::make_shared<Chromozome>(target, roi);
    for (int i = 0; i < Config::getParams().chromozome_length; ++i) ch->addRandomShape();

    ch->sort();

    return ch;
}


void Chromozome::sort ()
{
    // Sort the chromozome - put SMALL shapes to the top (this way they will not be covered by the big ones
    // when rendering)
    std::sort(this->_chromozome.begin(), this->_chromozome.end(),
              [](const std::shared_ptr<IShape> &s1, const std::shared_ptr<IShape> &s2) {
        return s1->getSizeGroup() < s2->getSizeGroup();
    });
}


size_t Chromozome::size () const
{
    return this->_chromozome.size();
}


void Chromozome::addRandomShape ()
{
    this->setDirty();

    switch (Config::getParams().shape_type) {
    case ShapeType::CIRCLE:
        this->_chromozome.push_back(Circle::randomCircle(this->_target));
        break;
    default:
        std::cout << "ERROR: Unknown shape type " << int(Config::getParams().shape_type) << std::endl;
        exit(EXIT_FAILURE);
        break;
    }
}


std::shared_ptr<IShape>& Chromozome::operator[] (size_t i)
{
    this->setDirty();

    assert(i < this->_chromozome.size());

    return this->_chromozome[i];
}


const std::shared_ptr<IShape>& Chromozome::operator[] (size_t i) const
{
    assert(i < this->_chromozome.size());

    return this->_chromozome[i];
}


double Chromozome::getFitness ()
{
    if (this->_dirty)
    {
        // The chromozome was editted - we need to recompute the fitness

        assert(this->_target->channels.size() == 3);
        assert(this->_target->channels[0].size() == this->_target->channels[1].size() && this->_target->channels[0].size() == this->_target->channels[2].size());

        // Render the image
        Renderer renderer(this->_target->image_size);
        std::vector<cv::Mat> channels = renderer.render(*this);

        // Compute pixel-wise difference
        this->_fitness = 0;
        for (size_t i = 0; i < 3; ++i)
        {
//            this->_fitness += computeDifference(this->_target->blurred_channels[i](this->_roi), channels[i](this->_roi), this->_target->weights(this->_roi));
            this->_fitness += computeDifference(this->_target->blurred_channels[i], channels[i], this->_target->weights);
        }

        // Just rendered and computed fitness
        this->_dirty = false;
    }

    return this->_fitness;
}


void Chromozome::setDirty ()
{
    this->_dirty = true;
}


void Chromozome::birthday ()
{
    this->_age++;
}


const std::shared_ptr<const Target>& Chromozome::getTarget () const
{
    return this->_target;
}


int Chromozome::getAge() const
{
    return this->_age;
}


cv::Mat Chromozome::asImage ()
{
    // Render the image represented by this chromozome
    Renderer r(this->_target->image_size);
    const std::vector<cv::Mat> channels = r.render(*this);

    // Merge the channels to a 3 channel cv::Mat
    cv::Mat image;
    cv::merge(channels, image);
    cv::cvtColor(image, image, CV_RGB2BGR);

//    cv::rectangle(image, this->_roi, cv::Scalar(0,0,255));

    return image;
}


void Chromozome::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


}
