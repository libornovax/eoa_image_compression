#include "Chromozome.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "Renderer.h"
#include "Config.h"
#include "shapes/Circle.h"


namespace eic {


Chromozome::Chromozome(const std::shared_ptr<const Target> &target)
    : _fitness(DBL_MAX),
      _dirty(true),
      _target(target)
{

}


std::shared_ptr<Chromozome> Chromozome::clone () const
{
    auto ch = std::make_shared<Chromozome>(this->_target);

    for (auto &shape: this->_chromozome)
    {
        ch->_chromozome.push_back(shape->clone());
    }

    return ch;
}


std::shared_ptr<Chromozome> Chromozome::randomChromozome (const std::shared_ptr<const Target> &target)
{
    auto ch = std::make_shared<Chromozome>(target);
    for (int i = 0; i < Config::getParams().chromozome_length; ++i) ch->addRandomShape();
    return ch;
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
        const std::vector<cv::Mat> channels = renderer.render(*this);

        // Compute pixel-wise difference
        this->_fitness = 0;
        for (size_t i = 0; i < 3; ++i)
        {
            cv::Mat diff;
            cv::subtract(this->_target->channels[i], channels[i], diff, cv::noArray(), CV_32FC1);
            cv::pow(diff, 2, diff);
            cv::Scalar total = cv::sum(diff);
            this->_fitness += total[0];
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


const std::shared_ptr<const Target>& Chromozome::getTarget () const
{
    return this->_target;
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

    return image;
}


void Chromozome::accept (IVisitor &visitor)
{
    visitor.visit(*this);
}


}
