//
// Libor Novak
// 10/11/2016
//

#ifndef IVISITOR_H
#define IVISITOR_H


namespace eic {

// Tell the compiler that the classes that we want to visit exist
class Circle;


class IVisitor
{
public:

    IVisitor () {}

    virtual void visit (const Circle &circle) = 0;

};


}


#endif // IVISITOR_H
