//
// Libor Novak
// 10/13/2016
//

#ifndef CONFIG_H
#define CONFIG_H

#include <string>


namespace eic {


enum class ShapeType
{
    CIRCLE = 1
};


enum class AlgorithmType
{
    HILL_CLIMBER = 1
};


struct MutatorParams
{
    double shape_mutation_prob;
};


struct HillClimberParams
{
    int num_iterations;
};


struct ConfigParams
{
    std::string path_image;
    int chromozome_length;
    ShapeType shape_type;
    AlgorithmType algorithm;

    MutatorParams mutator;
    HillClimberParams hill_climber;
};



class Config
{
public:

    static const ConfigParams& getParams ();

    static void loadParams (const std::string &path_config);


private:

    Config ();

    static Config& _getInstance ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    ConfigParams _params;

};


}


#endif // CONFIG_H
