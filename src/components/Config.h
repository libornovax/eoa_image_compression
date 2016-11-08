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
    HILL_CLIMBER            = 1,
    DIFFERENTIAL_EVOLUTION  = 2,
    CLASSIC_EA              = 3
};


struct MutatorParams
{
    double shape_mutation_prob;
    double shape_reorder_prob;

    double color_mutation_prob;
    double alpha_mutation_prob;
    double color_mutation_stddev;
    double alpha_mutation_stddev;

    double position_mutation_prob;
    double position_mutation_stddev;
    double position_reinitialization_prob;

    double radius_mutation_prob;
    double radius_mutation_sdtddev;
};


struct HillClimberParams
{
    int num_iterations;
    int pool_size;
};


struct DifferentialEvolutionParams
{
    int num_epochs;
    int population_size;
};


struct DifferentialCrossoverParams
{
    double shape_crossover_prob;
};


struct ClassicEAParams
{
    int num_epochs;
    int population_size;
};


struct ConfigParams
{
    std::string path_image;
    std::string path_out;
    int chromozome_length;
    ShapeType shape_type;
    AlgorithmType algorithm;

    MutatorParams mutator;
    HillClimberParams hill_climber;
    DifferentialEvolutionParams differential_evolution;
    DifferentialCrossoverParams differential_crossover;
    ClassicEAParams classic_ea;
};



class Config
{
public:

    /**
     * @brief Returns the loaded configuration parameters
     * @return Structure with config parameters
     */
    static const ConfigParams& getParams ();

    /**
     * @brief Loads parameters from a config YAML file
     * @param path_config Path to the config YAML file
     */
    static void loadParams (const std::string &path_config);

    /**
     * @brief Print the parameters to std::cout
     */
    static void print ();


private:

    Config ();

    static Config& _getInstance ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    ConfigParams _params;

};


}


#endif // CONFIG_H
