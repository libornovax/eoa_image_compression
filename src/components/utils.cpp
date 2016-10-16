#include "utils.h"


namespace eic {
namespace utils {


bool makeMutation (double p)
{
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(RGen::mt()) <= p;
}


}
}
