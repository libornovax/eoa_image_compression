//
// Libor Novak
// 10/11/2016
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>
#include <memory>
#include <random>
#include "shapes/IShape.h"


namespace eic {


// Chromozome is just a vector of shapes
typedef std::vector<std::shared_ptr<IShape>> Chromozome;


}


#endif // DEFINITIONS_H

