//
// Libor Novak
// 10/11/2016
//

#ifndef RENDERER_H
#define RENDERER_H

#include <opencv2/core/core.hpp>
#include <iostream>
#include "IVisitor.h"


namespace eic {


class Renderer : public IVisitor
{
public:

    /**
     * @brief Renderer
     * @param image_size Dimensions of the image to be rendered
     */
    Renderer (const cv::Size &image_size);


    /**
     * @brief Triggers rendering of the whole image and returns it
     * @param ch Chromozome (image representation) to be rendered
     * @return Vector of three CV_8UC1 matrices representing RGB channels of the image
     */
    const std::vector<cv::Mat> render (Chromozome &ch);

    /**
     * @brief Triggers the rendering of the whole chromozome
     * @param chromozome Chromozome to be rendered
     */
    virtual void visit (Chromozome &chromozome) override final;

    /**
     * @brief Renders a Circle into the current image
     * @param circle Circle to be rendered
     */
    virtual void visit (Circle &circle) override final;


private:

    /**
     * @brief Resets all channels of the image
     */
    void _reset ();

    /**
     * @brief Returns the rendered RGB channels
     * @return Vector of three CV_8UC1 matrices representing RGB channels of the image
     */
    std::vector<cv::Mat> _getRenderedChannels ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // RGB channels of the image
    std::vector<cv::Mat> _channels;
    cv::Size _image_size;

};


}


#endif // RENDERER_H
