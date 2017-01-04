#include "Chromozome.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Config.h"
#include "shapes/Circle.h"
#include "RGen.h"


namespace eic {


Chromozome::Chromozome(const std::shared_ptr<const Target> &target, const cv::Rect roi)
    : _fitness(DBL_MAX),
      _dirty(true),
      _target(target),
      _roi(roi),
      _roi_active(false)
{

}


std::shared_ptr<Chromozome> Chromozome::clone () const
{
    auto ch = std::make_shared<Chromozome>(this->_target, this->_roi);

    for (auto &shape: this->_chromozome)
    {
        ch->_chromozome.push_back(shape->clone());
    }

    ch->_fitness    = this->_fitness;
    ch->_dirty      = this->_dirty;
    ch->_roi_active = this->_roi_active;
    ch->_channels   = this->_channels;

    return ch;
}


void Chromozome::update (const std::shared_ptr<Chromozome> &other)
{
    this->_target     = other->_target;
    this->_roi        = other->_roi;
    this->_fitness    = other->_fitness;
    this->_dirty      = other->_dirty;
    this->_roi_active = other->_roi_active;
    this->_channels   = other->_channels;

    this->_chromozome.clear();
    for (auto &shape: other->_chromozome)
    {
        this->_chromozome.push_back(shape->clone());
    }
}


std::shared_ptr<Chromozome> Chromozome::randomChromozome (const std::shared_ptr<const Target> &target)
{
    // Select a random region of interest in target
    int square_dim = std::min(target->image_size.width, target->image_size.height) / 3;
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

    // WARNING! Has to be a stable sort because otherwise when we call this sort on an original and a clone
    // it gives different results (on Ubuntu with GCC!)
    std::stable_sort(this->_chromozome.begin(), this->_chromozome.end(),
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
    assert(!this->_dirty);

    return this->_fitness;
}


void Chromozome::setDirty ()
{
    this->_dirty = true;
}


const std::shared_ptr<const Target>& Chromozome::getTarget () const
{
    return this->_target;
}


void Chromozome::activateROI ()
{
    this->_roi_active = true;
    this->setDirty();
}


void Chromozome::deactivateROI ()
{
    this->_roi_active = false;
    this->setDirty();
}


cv::Mat Chromozome::asImage ()
{
    assert(!this->_dirty);

    // Merge the channels to a 3 channel cv::Mat
    cv::Mat image;
    cv::merge(this->_channels, image);
    cv::cvtColor(image, image, CV_RGB2BGR);

    if (this->_roi_active) cv::rectangle(image, this->_roi, cv::Scalar(0,0,255));

    return image;
}


std::vector<cv::Mat>& Chromozome::channels ()
{
    return this->_channels;
}


bool Chromozome::roiActive () const
{
    return this->_roi_active;
}


const cv::Rect& Chromozome::getROI () const
{
    return this->_roi;
}


void Chromozome::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


}
