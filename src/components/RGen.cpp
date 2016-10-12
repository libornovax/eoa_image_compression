#include "RGen.h"


namespace eic {


std::mt19937& RGen::mt ()
{
    static RGen rg;

    return rg._mt;
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

RGen::RGen()
    : _mt(_rd())
{
}


}
