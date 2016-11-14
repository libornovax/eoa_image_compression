//
// Libor Novak
// 11/14/2016
//

#ifndef STATS_H
#define STATS_H

#include <vector>


namespace eic {


class Stats
{
public:

    Stats ();

    /**
     * @brief Add a new entry to the statistics
     * @param epoch
     * @param best_fitness
     * @param worst_fitness
     * @param mean_fitness
     * @param stddev_fitness
     */
    void add (int epoch, double best_fitness, double worst_fitness, double mean_fitness=0.0, double stddev_fitness=0.0);

    /**
     * @brief Saves the statistics to statistics.txt into the output folder
     */
    void save ();


private:

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    std::vector<int>    _epoch;
    std::vector<double> _best_fitness;
    std::vector<double> _worst_fitness;
    std::vector<double> _mean_fitness;
    std::vector<double> _stddev_fitness;

};


}


#endif // STATS_H
