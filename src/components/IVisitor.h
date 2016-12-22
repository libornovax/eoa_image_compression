//
// Libor Novak
// 10/11/2016
//

#ifndef IVISITOR_H
#define IVISITOR_H


namespace eic {

// Tell the compiler that the classes that we want to visit exist
class Chromozome;
class Circle;
class Rectangle;


class IVisitor
{
public:

    IVisitor () {}

    virtual void visit (Chromozome &chromozome) = 0;
    virtual void visit (Circle &circle) = 0;
    virtual void visit (Rectangle &rect) = 0;

};


}


#endif // IVISITOR_H
