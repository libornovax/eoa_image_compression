#include "Stats.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include "components/Config.h"


namespace eic {


Stats::Stats ()
{

}


void Stats::add (int epoch, double best_fitness, double worst_fitness, double mean_fitness, double stddev_fitness)
{
    this->_epoch.push_back(epoch);
    this->_best_fitness.push_back(best_fitness);
    this->_worst_fitness.push_back(worst_fitness);
    this->_mean_fitness.push_back(mean_fitness);
    this->_stddev_fitness.push_back(stddev_fitness);
}


void Stats::save ()
{
    assert(this->_epoch.size() == this->_best_fitness.size() && this->_epoch.size() == this->_mean_fitness.size());

    std::string path = Config::getParams().path_out + "/statistics.txt";
    std::cout << "Saving statistics to file: " << path << std::endl;

    std::ofstream outfile(path);
    if (outfile)
    {
        // Write the header
        outfile << "epoch best_fitness worst_fitness mean_fitness stddev_fitness" << std::endl;

        for (int i = 0; i < this->_epoch.size(); ++i)
        {
            outfile << this->_epoch[i] << " ";
            outfile << this->_best_fitness[i] << " ";
            outfile << this->_worst_fitness[i] << " ";
            outfile << this->_mean_fitness[i] << " ";
            outfile << this->_stddev_fitness[i] << std::endl;
        }

        outfile.close();
    }
}


}
