//
// Libor Novak
// 12/11/2016
//

#ifndef HILLCLIMBERPOOL_H
#define HILLCLIMBERPOOL_H

#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "components/Chromozome.h"


namespace eic {


/**
 * @brief The HillClimberPool class
 * Pool of threads runing the hill climber algorithm - faster optimization on computers with multiple core
 * processors
 */
class HillClimberPool
{
public:

    HillClimberPool (int pool_size);
    ~HillClimberPool ();

    /**
     * @brief Shuts down the workers. Does NOT wait until the whole queue is processed!
     */
    void shutDown ();

    /**
     * @brief Adds a chromozome to the queue for processing
     * @param ch Chromozome
     */
    void addChromozome (const std::shared_ptr<Chromozome> &ch);

    /**
     * @brief Waits for the queue processing to finish (does not return untill the input queue is empty)
     */
    void waitToFinish ();


private:

    /**
     * @brief Starts the hill climber threads
     */
    void _launch ();

    /**
     * @brief Worker that will be processing the queue
     */
    void _workerThread ();


    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    // Threads that run the hill climber algorithm
    std::vector<std::thread> _worker_pool;
    // Number of currently running workers
    std::atomic<int> _num_running_workers;
    // Number of workers that are currently processing something
    std::atomic<int> _num_processing_workers;
    // Input queue of chromozomes
    std::deque<std::shared_ptr<Chromozome>> _queue;
    // Maximum size of the input queue
    int _queue_size;

    // Queue access logic
    std::mutex _mtx;
    std::condition_variable _cv_full;
    std::condition_variable _cv_empty;
    std::atomic<bool> _shut_down;
    // Signal that one chromozome has been processed (for waitToFinish())
    std::condition_variable _cv_one_done;

};


}


#endif // HILLCLIMBERPOOL_H
