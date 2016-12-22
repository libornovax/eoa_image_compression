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
    CIRCLE    = 1,
    RECTANGLE = 2
};


enum class AlgorithmType
{
    HILL_CLIMBER      = 1,
    CLASSIC_EA        = 2,
    STEADY_STATE_EA   = 3,
    INTERLEAVED_EA    = 4
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
    double radius_mutation_stddev;

    double wh_mutation_prob;
    double wh_mutation_stddev;
};


struct HillClimberParams
{
    int num_iterations;
    int pool_size;
};


struct InterleavedEAParams
{
    int hillclimb_frequency;
};


struct ClassicEAParams
{
    int num_epochs;
    int population_size;
    int tournament_size;
    double best_selection_prob;
    double crossover_prob;

    InterleavedEAParams interleaved_ea;
};


struct ConfigParams
{
    std::string path_image;
    std::string path_image_weights;
    double max_weight;
    std::string path_out;
    int chromozome_length;
    ShapeType shape_type;
    AlgorithmType algorithm;

    MutatorParams mutator;
    HillClimberParams hill_climber;
    ClassicEAParams ea;
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
