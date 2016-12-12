#include "ClassicEA.h"

#include <random>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "entities/Mutator.h"
#include "components/Config.h"
#include "components/utils.h"
#include "shapes/Circle.h"


namespace eic {


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
    for (int e = 0; e < Config::getParams().ea.num_epochs; ++e)
    {
        this->_stats.add(e, this->_best_chromozome->getFitness(), this->_worst_chromozome->getFitness(),
                         ClassicEA::_meanFitness(this->_population), ClassicEA::_stddevFitness(this->_population));

        if (e % 50 == 0)
        {
            this->_saveCurrentPopulation(e);
            this->_stats.save();
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
            if (utils::makeMutation(Config::getParams().ea.crossover_prob))
            {
                // Do it multiple times to exchange a larger part of the chromozome
                ClassicEA::_onePointCrossover(offspring1, offspring2);
                ClassicEA::_onePointCrossover(offspring1, offspring2);
            }

            // Mutation
            offspring1->accept(mutator);
            offspring2->accept(mutator);

            // Put the offspring into the new population
            new_population[2*i+1] = offspring1;
            new_population[2*i+2] = offspring2;
        }

        // Generational replacement with elitism (elitism is already taken care of)
        this->_population = new_population;

        // Sort the population by fitness
        ClassicEA::_sortPopulation(this->_population);

        this->_updateBestChromozome(e);
        this->_updateWorstChromozome(e);
    }

    return this->_best_chromozome->clone();
}


// -----------------------------------------  PROTECTED METHODS  ----------------------------------------- //

void ClassicEA::_initializePopulation ()
{
    // Generate random chromozomes
    for (int i = 0; i < Config::getParams().ea.population_size; ++i)
    {
        this->_population.push_back(Chromozome::randomChromozome(this->_target));
    }

    ClassicEA::_sortPopulation(this->_population);

    // Just set the best as a random one from the population
    this->_best_chromozome = this->_population[0]->clone();
    this->_worst_chromozome = this->_population.back()->clone();

    {
        cv::Mat image = this->_best_chromozome->asImage();
        cv::imwrite(eic::Config::getParams().path_out + "/approx_0.png", image);
    }
}


void ClassicEA::_initializeNewPopulation (std::vector<std::shared_ptr<Chromozome>> &new_population) const
{
    new_population.resize(this->_population.size());

    // Elitism
    new_population[0] = this->_best_chromozome->clone();

    // This is just because we need to add even number of chromozomes with crossover (there has to be an even
    // number free spots). That means that if it is odd we need to add one more chromozome
    if (this->_population.size() % 2 == 0)
    {
        new_population[this->_population.size()-1] = this->_population[this->_tournamentSelection()]->clone();
    }
}


void ClassicEA::_updateBestChromozome (int e)
{
    // WARNING! We suppose the population is sorted from best to worst!!
    if (this->_population[0]->getFitness() < this->_best_chromozome->getFitness())
    {
        std::cout << "[" << e << "] New best difference: " << this->_population[0]->getFitness() << std::endl;

        // Save the current best image
        if (e-this->_last_save > 50)
        {
            this->_last_save = e;
            cv::Mat image = this->_population[0]->asImage();
            cv::imwrite(Config::getParams().path_out + "/approx_" + std::to_string(e) + ".png", image);
        }
    }

    this->_best_chromozome = this->_population[0]->clone();
}


void ClassicEA::_updateWorstChromozome (int e)
{
    // WARNING! We suppose the population is sorted from best to worst!!
    this->_worst_chromozome = this->_population.back()->clone();
}


int ClassicEA::_tournamentSelection(int exclude_idx) const
{
    // Select n random individuals for the tournament and select the best one from them
    std::uniform_int_distribution<int> dist(0, this->_population.size()-1);

    // Select n random individuals
    std::vector<std::pair<int, double>> selected;
    while (selected.size() < Config::getParams().ea.tournament_size)
    {
        int i = dist(RGen::mt());

        // We do not want this index in the list
        if (i == exclude_idx) continue;

        // Check if this index is already in the selection, if not add it
        if (std::find_if(selected.begin(), selected.end(),
                         [&i](const std::pair<int, double> &p) { return p.first == i; }) == selected.end())
        {
            selected.emplace_back(i, this->_population[i]->getFitness());
        }
    }

    // Order them by ascending fitness
    std::sort(selected.begin(), selected.end(),
              [](const std::pair<int, double> &a, const std::pair<int, double> &b){ return a.second < b.second; });

    for (auto sel: selected)
    {
        if (utils::makeMutation(Config::getParams().ea.best_selection_prob))
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

    // Select a random position and radius that will initialize the crossover position. The position is
    // selected as the center of a random small or medium shape
    cv::Point position = utils::selectRandomPositionForCrossover(offspring1, offspring2);
    std::uniform_int_distribution<int> distr(20, this->_target->image_size.width/8);
    int radius = distr(RGen::mt());

    // Find all shapes in chromozomes offspring1 and offspring2 that intersect this shape
    std::vector<int> idxs_i1 = utils::findIntersectingShapesIdxs(position, radius, offspring1);
    std::vector<int> idxs_i2 = utils::findIntersectingShapesIdxs(position, radius, offspring2);

//    {
//        cv::Size image_size = this->_target->image_size;
//        cv::Mat canvas(image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_i1.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(offspring1->operator [](idxs_i1[i]));
//            cv::circle(canvas, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::circle(canvas, position, radius, cv::Scalar(0,0,255), 1);
//        cv::Mat canvas2(image_size, CV_8UC3, cv::Scalar(255,255,255));
//        for (int i = idxs_i2.size()-1; i >= 0; --i)
//        {
//            auto circ = std::static_pointer_cast<Circle>(offspring2->operator [](idxs_i2[i]));
//            cv::circle(canvas2, circ->getCenter(), circ->getRadius(), cv::Scalar(circ->getB(), circ->getG(), circ->getR()), -1);
//        }
//        cv::circle(canvas2, position, radius, cv::Scalar(0,0,255), 1);
//        cv::imshow("crossover offspring1", canvas);
//        cv::imshow("crossover offspring2", canvas2);
//        std::cout << "Crossover size: " << idxs_i1.size() << "  " << idxs_i2.size() << std::endl;
//        cv::waitKey();

//        cv::imshow("offspring1 before", offspring1->asImage());
//        cv::imshow("offspring2 before", offspring2->asImage());
//    }

    // Exchange those shapes (or parts of them)
    for (int i = 0; i < idxs_i1.size() && i < idxs_i2.size(); ++i)
    {
        // Exchange the shapes only if they are of the same size groups - we do not want to mix them because
        // then we start losing the small ones
        if ((offspring1->operator [](idxs_i1[i])->getSizeGroup() == SizeGroup::SMALL && offspring2->operator [](idxs_i2[i])->getSizeGroup() == SizeGroup::SMALL) ||
            (offspring1->operator [](idxs_i1[i])->getSizeGroup() == SizeGroup::MEDIUM && offspring2->operator [](idxs_i2[i])->getSizeGroup() == SizeGroup::MEDIUM) ||
            (offspring1->operator [](idxs_i1[i])->getSizeGroup() == SizeGroup::LARGE && offspring2->operator [](idxs_i2[i])->getSizeGroup() == SizeGroup::LARGE))
        {
            auto tmp = offspring1->operator [](idxs_i1[i]);
            offspring1->operator [](idxs_i1[i]) = offspring2->operator [](idxs_i2[i]);
            offspring2->operator [](idxs_i2[i]) = tmp;
        }
    }

    // If we keep the size groups we do not need the sort here
//    offspring1->sort();
//    offspring2->sort();

//    {
//        cv::imshow("offspring1 after", offspring1->asImage());
//        cv::imshow("offspring2 after", offspring2->asImage());
//        cv::waitKey();
//    }
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

    cv::imwrite(Config::getParams().path_out + "/population_" + std::to_string(epoch) + ".png", canvas);
}


void ClassicEA::_sortPopulation (std::vector<std::shared_ptr<Chromozome>> &population)
{
    // Sort the population by ascending fitness
    std::sort(population.begin(), population.end(),
              [] (const std::shared_ptr<Chromozome> &ch1, const std::shared_ptr<Chromozome> &ch2) {
        return ch1->getFitness() < ch2->getFitness();
    });
}


double ClassicEA::_meanFitness (const std::vector<std::shared_ptr<Chromozome>> &population)
{
    double sum = 0.0;

    for (auto &ch: population) sum += ch->getFitness();

    return sum / population.size();
}


double ClassicEA::_stddevFitness (const std::vector<std::shared_ptr<Chromozome>> &population)
{
    double mean = ClassicEA::_meanFitness(population);
    double sum = 0.0;

    for (auto &ch: population)
    {
        sum += (ch->getFitness()-mean)*(ch->getFitness()-mean);
    }

    return std::sqrt(sum / population.size());
}


}
