//
// Libor Novak
// 10/11/2016
//

#ifndef RENDERER_H
#define RENDERER_H

#include <opencv2/core/core.hpp>
#include <iostream>
#include "definitions.h"
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
     * @brief Triggers rendering of the whole image from the given Chromozome
     * @param ch Chromozome (image representation) to be rendered
     * @return 3 channels representing the rendered image
     */
    const std::vector<cv::Mat> render (const Chromozome &ch);

    /**
     * @brief Renders a Circle into the current image
     * @param circle Circle to be rendered
     */
    virtual void visit (const Circle &circle) override final;


private:

    /**
     * @brief Resets all channels of the image
     */
    void _reset ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // RGB channels of the image
    std::vector<cv::Mat> _channels;
    cv::Size _image_size;

};


}


#endif // RENDERER_H
