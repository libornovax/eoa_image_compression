#include "Config.h"

#include <iostream>
#include "yaml-cpp/yaml.h"


namespace eic {


const ConfigParams& Config::getParams ()
{
    return Config::_getInstance()._params;
}


void Config::loadParams (const std::string &path_config)
{
    std::cout << "Loading config: " << path_config << std::endl;
    YAML::Node config = YAML::LoadFile(path_config);

    // General parameters
    Config::_getInstance()._params.path_image = config["path_image"].as<std::string>();
    Config::_getInstance()._params.path_image_weights = config["path_image_weights"].as<std::string>();
    Config::_getInstance()._params.max_weight = config["max_weight"].as<double>();
    Config::_getInstance()._params.path_out = config["path_out"].as<std::string>();
    Config::_getInstance()._params.chromozome_length = config["chromozome_length"].as<int>();
    Config::_getInstance()._params.shape_type = ShapeType(config["shape_type"].as<int>());
    Config::_getInstance()._params.algorithm = AlgorithmType(config["algorithm"].as<int>());

    // Mutator settings
    YAML::Node mutator = config["mutator"];
    Config::_getInstance()._params.mutator.shape_mutation_prob = mutator["shape_mutation_prob"].as<double>();
    Config::_getInstance()._params.mutator.shape_reorder_prob = mutator["shape_reorder_prob"].as<double>();
    Config::_getInstance()._params.mutator.color_mutation_prob = mutator["color_mutation_prob"].as<double>();
    Config::_getInstance()._params.mutator.alpha_mutation_prob = mutator["alpha_mutation_prob"].as<double>();
    Config::_getInstance()._params.mutator.color_mutation_stddev = mutator["color_mutation_stddev"].as<double>();
    Config::_getInstance()._params.mutator.alpha_mutation_stddev = mutator["alpha_mutation_stddev"].as<double>();
    Config::_getInstance()._params.mutator.position_mutation_prob = mutator["position_mutation_prob"].as<double>();
    Config::_getInstance()._params.mutator.position_mutation_stddev = mutator["position_mutation_stddev"].as<double>();
    Config::_getInstance()._params.mutator.position_reinitialization_prob = mutator["position_reinitialization_prob"].as<double>();
    Config::_getInstance()._params.mutator.radius_mutation_prob = mutator["radius_mutation_prob"].as<double>();
    Config::_getInstance()._params.mutator.radius_mutation_stddev = mutator["radius_mutation_sdtddev"].as<double>();

    // ClassicEA settings
    if (Config::_getInstance()._params.algorithm == AlgorithmType::CLASSIC_EA ||
            Config::_getInstance()._params.algorithm == AlgorithmType::STEADY_STATE_EA ||
            Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        YAML::Node ea = config["ea"];
        Config::_getInstance()._params.ea.num_epochs = ea["num_epochs"].as<int>();
        Config::_getInstance()._params.ea.population_size = ea["population_size"].as<int>();
        Config::_getInstance()._params.ea.tournament_size = ea["tournament_size"].as<int>();
        Config::_getInstance()._params.ea.best_selection_prob = ea["best_selection_prob"].as<double>();
        Config::_getInstance()._params.ea.crossover_prob = ea["crossover_prob"].as<double>();

        if (Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
        {
            YAML::Node iea = ea["interleaved_ea"];
            Config::_getInstance()._params.ea.interleaved_ea.hillclimb_frequency = iea["hillclimb_frequency"].as<int>();
        }
    }

    // HillClimber settings
    if (Config::_getInstance()._params.algorithm == AlgorithmType::HILL_CLIMBER ||
            Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        YAML::Node hc = config["hill_climber"];
        Config::_getInstance()._params.hill_climber.num_iterations = hc["num_iterations"].as<int>();
        Config::_getInstance()._params.hill_climber.pool_size = hc["pool_size"].as<int>();
    }
}


void Config::print ()
{
    std::cout << "============================================================================" << std::endl;
    std::cout << "========================  CONFIGURATION PARAMETERS  ========================" << std::endl;
    std::cout << "============================================================================" << std::endl;

    std::cout << "path_image:                     " << Config::_getInstance()._params.path_image << std::endl;
    std::cout << "path_image_weights:             " << Config::_getInstance()._params.path_image_weights << std::endl;
    std::cout << "max_weight:                     " << Config::_getInstance()._params.max_weight << std::endl;
    std::cout << "path_out:                       " << Config::_getInstance()._params.path_out << std::endl;
    std::cout << "chromozome_length:              " << Config::_getInstance()._params.chromozome_length << std::endl;
    std::cout << "shape_type:                     " << int(Config::_getInstance()._params.shape_type) << std::endl;
    std::cout << "algorithm:                      " << int(Config::_getInstance()._params.algorithm) << std::endl;

    std::cout << "================================  MUTATOR  =================================" << std::endl;

    std::cout << "shape_mutation_prob:            " << Config::_getInstance()._params.mutator.shape_mutation_prob << std::endl;
    std::cout << "shape_reorder_prob:             " << Config::_getInstance()._params.mutator.shape_reorder_prob << std::endl;
    std::cout << "color_mutation_prob:            " << Config::_getInstance()._params.mutator.color_mutation_prob << std::endl;
    std::cout << "alpha_mutation_prob:            " << Config::_getInstance()._params.mutator.alpha_mutation_prob << std::endl;
    std::cout << "color_mutation_stddev:          " << Config::_getInstance()._params.mutator.color_mutation_stddev << std::endl;
    std::cout << "alpha_mutation_stddev:          " << Config::_getInstance()._params.mutator.alpha_mutation_stddev << std::endl;
    std::cout << "position_mutation_prob:         " << Config::_getInstance()._params.mutator.position_mutation_prob << std::endl;
    std::cout << "position_mutation_stddev:       " << Config::_getInstance()._params.mutator.position_mutation_stddev << std::endl;
    std::cout << "position_reinitialization_prob: " << Config::_getInstance()._params.mutator.position_reinitialization_prob << std::endl;
    std::cout << "radius_mutation_prob:           " << Config::_getInstance()._params.mutator.radius_mutation_prob << std::endl;
    std::cout << "radius_mutation_sdtddev:        " << Config::_getInstance()._params.mutator.radius_mutation_stddev << std::endl;

    if (Config::_getInstance()._params.algorithm == AlgorithmType::CLASSIC_EA)
    {
        std::cout << "=====================  CLASSIC EVOLUTIONARY ALGORITHM  =====================" << std::endl;
    }
    else if (Config::_getInstance()._params.algorithm == AlgorithmType::STEADY_STATE_EA)
    {
        std::cout << "==================  STEADY STATE EVOLUTIONARY ALGORITHM  ===================" << std::endl;
    }
    else if (Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        std::cout << "==================  INTERLEAVED EVOLUTIONARY ALGORITHM  ====================" << std::endl;
    }
    if (Config::_getInstance()._params.algorithm == AlgorithmType::CLASSIC_EA ||
            Config::_getInstance()._params.algorithm == AlgorithmType::STEADY_STATE_EA ||
            Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        std::cout << "num_epochs:                     " << Config::_getInstance()._params.ea.num_epochs << std::endl;
        std::cout << "population_size:                " << Config::_getInstance()._params.ea.population_size << std::endl;
        std::cout << "tournament_size:                " << Config::_getInstance()._params.ea.tournament_size << std::endl;
        std::cout << "best_selection_prob:            " << Config::_getInstance()._params.ea.best_selection_prob << std::endl;
        std::cout << "crossover_prob:                 " << Config::_getInstance()._params.ea.crossover_prob << std::endl;
    }
    if (Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        std::cout << "hillclimb_frequency:            " << Config::_getInstance()._params.ea.interleaved_ea.hillclimb_frequency << std::endl;
    }

    if (Config::_getInstance()._params.algorithm == AlgorithmType::HILL_CLIMBER ||
            Config::_getInstance()._params.algorithm == AlgorithmType::INTERLEAVED_EA)
    {
        std::cout << "==============================  HILL CLIMBER  ==============================" << std::endl;
        std::cout << "num_iterations:                 " << Config::_getInstance()._params.hill_climber.num_iterations << std::endl;
        std::cout << "pool_size:                      " << Config::_getInstance()._params.hill_climber.pool_size << std::endl;
    }


    std::cout << "============================================================================" << std::endl << std::endl;
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

Config::Config ()
{
}


Config& Config::_getInstance ()
{
    static Config instance;
    return instance;
}


}
