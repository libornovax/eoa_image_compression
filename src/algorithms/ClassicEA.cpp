#include "ClassicEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"


namespace eic {

namespace {

    /**
     * @brief Finds indices of shapes in the chromozome, which contain the given point
     * @param p Point
     * @param chromozome
     * @return Vector of indices in the chromozome
     */
    std::vector<int> findContainingShapesIdxs (const cv::Point &p, const std::shared_ptr<Chromozome> &chromozome)
    {
        std::vector<int> containing_idxs;

        for (int i = 0; i < chromozome->size(); ++i)
        {
            if (chromozome->operator [](i)->contains(p))
            {
                containing_idxs.push_back(i);
            }
        }

        return containing_idxs;
    }

}


ClassicEA::ClassicEA (const std::shared_ptr<const Target> &target)
    : _target(target),
      _last_save(0)
{

}


std::shared_ptr<Chromozome> ClassicEA::run ()
{
    this->_initializePopulation();

    // Run the evolution
    Mutator mutator(this->_target->image_size);
    for (int e = 0; e < Config::getParams().classic_ea.num_epochs; ++e)
    {
        if (e % 10 == 0)
        {
            this->_saveCurrentPopulation(e);
        }

        std::vector<std::shared_ptr<Chromozome>> new_population;
        this->_initializeNewPopulation(new_population);

        // -- EVOLUTION -- //
        // Evolve each individual in the population
        for (int i = 0; i < int(this->_population.size()-1)/2; ++i)
        {
            // Tournament selection
            // Select 2 individuals for crossover
            int i1 = this->_tournamentSelection();
            int i2 = this->_tournamentSelection(i1);

            // Careful! We have to clone here!!!
            auto offspring1 = this->_population[i1]->clone();
            auto offspring2 = this->_population[i2]->clone();

            // Crossover
            if (utils::makeMutation(Config::getParams().classic_ea.crossover_prob))
            {
                ClassicEA::_onePointCrossover(offspring1, offspring2);
            }

            // Mutation
            offspring1->accept(mutator);
            offspring2->accept(mutator);

            // Put the offspring into the new population
            new_population[2*i+1] = offspring1;
            new_population[2*i+2] = offspring2;
        }

        // Sort the population by fitness
        std::sort(new_population.begin(), new_population.end(),
                  [] (const std::shared_ptr<Chromozome> &ch1, const std::shared_ptr<Chromozome> &ch2) {
            return ch1->getFitness() < ch2->getFitness();
        });


        this->_updateBestChromozome(new_population, e);

        // Replace some of the individuals with random new ones to keep diversity in the population
        this->_refreshPopulation(new_population);

        // Generational replacement with elitism (elitism is already taken care of)
        this->_population = new_population;
    }

    return this->_best_chromozome->clone();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void ClassicEA::_initializePopulation ()
{
    // Generate random chromozomes
    for (int i = 0; i < Config::getParams().classic_ea.population_size; ++i)
    {
        this->_population.push_back(Chromozome::randomChromozome(this->_target));
    }

    // Just set the best as a random one from the population
    this->_best_chromozome = this->_population[0];

    {
        cv::Mat image = this->_best_chromozome->asImage();
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }
}


void ClassicEA::_initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const
{
    new_population.resize(this->_population.size());

    // Elitism
    new_population[0] = this->_best_chromozome;

    // This is just because we need to add even number of chromozomes with crossover (there has to be an even
    // number free spots). That means that if it is odd we need to add one more chromozome
    if (this->_population.size() % 2 == 0)
    {
        new_population[this->_population.size()-1] = this->_population[this->_tournamentSelection()]->clone();
    }
}


void ClassicEA::_updateBestChromozome (const std::vector<std::shared_ptr<Chromozome>> &new_population, int e)
{
    // WARNING! We suppose the population is sorted from best to worst!!

    if (new_population[0]->getFitness() < this->_best_chromozome->getFitness())
    {
        std::cout << "[" << e << "] New best difference: " << new_population[0]->getFitness() << std::endl;
        this->_best_chromozome = new_population[0];

        // Save the current best image
        if (e-this->_last_save > 100)
        {
            this->_last_save = e;
            cv::Mat image = this->_best_chromozome->asImage();
            cv::imwrite(eic::Config::getParams().path_out + "/approx_" + std::to_string(e) + ".png", image);
        }
    }
}


void ClassicEA::_refreshPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const
{
    // Replace every n-th chromozome with a new one
    for (int i = 1; i < new_population.size(); i+=5)
    {
        new_population[i] = Chromozome::randomChromozome(this->_target);
    }
}


int ClassicEA::_tournamentSelection(int exclude_idx) const
{
    // Select n random individuals for the tournament and select the best one from them
    // We imitate selecting n individuals by shuffling the indices in the population and taking the first
    // n individuals

    // Vector 0, 1, 2, ...
    std::vector<int> idxs(this->_population.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    // Erase the index we want to exclude
    if (exclude_idx >= 0 && exclude_idx < idxs.size())
    {
        idxs.erase(idxs.begin()+exclude_idx);
    }

    std::random_shuffle(idxs.begin(), idxs.end());

    // Take the first tournament_size indices
    std::vector<std::pair<int, double>> selected;
    for (int i = 0; i < Config::getParams().classic_ea.tournament_size; ++i)
    {
        selected.emplace_back(idxs[i], this->_population[idxs[i]]->getFitness());
    }

    // Order them by ascending difference
    std::sort(selected.begin(), selected.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b){ return a.second < b.second; });

    for (auto sel: selected)
    {
        if (utils::makeMutation(0.5))
        {
            return sel.first;
        }
    }

    return selected.back().first;
}


void ClassicEA::_onePointCrossover (std::shared_ptr<Chromozome> &offspring1, std::shared_ptr<Chromozome> &offspring2)
{
    // The "One Point" is a little misleading in the name. The crossover works as follows: We select a random
    // point in the image (random coordinates) and select all shapes that intersect the given point. Then we
    // exchange those shapes between the chromozomes.

    // We need to trigger fitness recomputation after crossover
    offspring1->setDirty();
    offspring2->setDirty();

    // Select a random position in the image
    cv::Size image_size = offspring1->getTarget()->image_size;
    std::uniform_int_distribution<int> distx(0, image_size.width);
    std::uniform_int_distribution<int> disty(0, image_size.height);

    cv::Point position(distx(RGen::mt()), disty(RGen::mt()));

    // Find all shapes in chromozomes i1 and i2 that contain this position
    std::vector<int> idxs_i1 = findContainingShapesIdxs(position, offspring1);
    std::vector<int> idxs_i2 = findContainingShapesIdxs(position, offspring2);

//    {
//        cv::Mat canvas(this->_image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_x.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(this->_x->operator [](idxs_x[i]));
//            cv::circle(canvas, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::Mat canvas2(this->_image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_ch.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(chromozome[idxs_ch[i]]);
//            cv::circle(canvas2, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::imshow("crossover x", canvas);
//        cv::imshow("crossover ch", canvas2);
//        std::cout << "Crossover size: " << idxs_x.size() << "  " << idxs_ch.size() << std::endl;
//        cv::waitKey();
//    }

    // Exchange those shapes (or parts of them)
    for (int i = 0; i < idxs_i1.size() && i < idxs_i2.size(); ++i)
    {
        auto tmp = offspring1->operator [](idxs_i1[i]);
        offspring1->operator [](idxs_i1[i]) = offspring2->operator [](idxs_i2[i]);
        offspring2->operator [](idxs_i2[i]) = tmp;
    }
}


void ClassicEA::_saveCurrentPopulation (int epoch)
{
    // Spacing between the images in the grid
    const int grid_spacing = 10;

    const int grid_dim = std::ceil(std::sqrt(double(this->_population.size())));
    const int cell_width = this->_target->image_size.width + grid_spacing;
    const int cell_height = this->_target->image_size.height + grid_spacing;

    cv::Mat canvas(grid_dim*cell_height+grid_spacing, grid_dim*cell_width+grid_spacing,
                   CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < this->_population.size(); ++i)
    {
        int x = i % grid_dim;
        int y = i / grid_dim;

        cv::Rect roi(x*cell_width+grid_spacing, y*cell_height+grid_spacing,
                     this->_target->image_size.width, this->_target->image_size.height);
        cv::Mat canvas_crop = canvas(roi);

        // Copy the rendered image into the grid
        this->_population[i]->asImage().copyTo(canvas_crop);
    }

    cv::imwrite(eic::Config::getParams().path_out + "/population_" + std::to_string(epoch) + ".png", canvas);
}


}
