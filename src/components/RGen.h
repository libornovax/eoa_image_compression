//
// Libor Novak
// 10/12/2016
//

#ifndef RGEN_H
#define RGEN_H

#include <random>


namespace eic {


/**
 * @brief A random number generator
 * We need a singleton to initialize the std random number generator, which we will then use everywhere in
 * the code to produce pseudorandom numbers
 */
class RGen
{
public:

    /**
     * @brief Returns a random generator, which is already seeded with a random_device
     */
    static std::mt19937& mt ();


private:

    // This is a singleton
    RGen ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    std::random_device _rd;
    std::mt19937 _mt;

};


}


#endif // RGEN_H
