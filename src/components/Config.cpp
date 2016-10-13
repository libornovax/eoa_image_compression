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
    Config::_getInstance()._params.path_out = config["path_out"].as<std::string>();
    Config::_getInstance()._params.chromozome_length = config["chromozome_length"].as<int>();
    Config::_getInstance()._params.shape_type = ShapeType(config["shape_type"].as<int>());
    Config::_getInstance()._params.algorithm = AlgorithmType(config["algorithm"].as<int>());

    // Mutator settings
    YAML::Node mutator = config["mutator"];
    Config::_getInstance()._params.mutator.shape_mutation_prob = mutator["shape_mutation_prob"].as<double>();

    // HillClimber settings
    YAML::Node hc = config["hill_climber"];
    Config::_getInstance()._params.hill_climber.num_iterations = hc["num_iterations"].as<int>();
}


void Config::print ()
{
    std::cout << "============================================================================" << std::endl;
    std::cout << "========================  CONFIGURATION PARAMETERS  ========================" << std::endl;
    std::cout << "============================================================================" << std::endl;

    std::cout << "path_image:                    " << Config::_getInstance()._params.path_image << std::endl;
    std::cout << "path_out:                      " << Config::_getInstance()._params.path_out << std::endl;
    std::cout << "chromozome_length:             " << Config::_getInstance()._params.chromozome_length << std::endl;
    std::cout << "shape_type:                    " << int(Config::_getInstance()._params.shape_type) << std::endl;
    std::cout << "algorithm:                     " << int(Config::_getInstance()._params.algorithm) << std::endl;

    std::cout << "================================  MUTATOR  =================================" << std::endl;

    std::cout << "shape_mutation_prob:           " << Config::_getInstance()._params.mutator.shape_mutation_prob << std::endl;


    if (Config::_getInstance()._params.algorithm == AlgorithmType::HILL_CLIMBER)
    {
        std::cout << "==============================  HILL CLIMBER  ==============================" << std::endl;
        std::cout << "num_iterations:                " << Config::_getInstance()._params.hill_climber.num_iterations << std::endl;
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
