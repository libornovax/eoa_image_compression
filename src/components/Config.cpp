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
