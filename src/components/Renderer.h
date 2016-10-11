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

    Renderer (const cv::Size &size);


    const std::vector<cv::Mat> render (const Chromozome &ch);

    virtual void visit (const Circle &circle) override final;

private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // RGB channels of the image
    std::vector<cv::Mat> _channels;

};


}


#endif // RENDERER_H
