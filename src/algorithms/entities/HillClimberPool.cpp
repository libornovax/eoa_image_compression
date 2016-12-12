#include "HillClimberPool.h"

#include "algorithms/HillClimber.h"
#include "components/Config.h"


namespace eic {

namespace {

    /**
     * @brief Prints text to console without being interrupted by another syncPrint
     * @param txt Text to print
     */
    void syncPrint (const std::string &txt)
    {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lk(mtx);
        std::cout << txt << std::endl;
    }
}


HillClimberPool::HillClimberPool (int queue_size)
    : _shut_down(true),
      _queue_size(queue_size),
      _num_running_workers(0)
{
    this->_launch();
}


HillClimberPool::~HillClimberPool ()
{
    if (!this->_shut_down) this->shutDown();
}


void HillClimberPool::shutDown()
{
    this->_shut_down = true;

    // If there are waiting threads we need to wake them up and shut down
    this->_cv_empty.notify_all();

    // Wait for the workers to finish
    for (auto &worker: this->_worker_pool)
    {
        worker.join();
    }

    assert(this->_num_running_workers == 0);
}


void HillClimberPool::addChromozome (const std::shared_ptr<Chromozome> &ch)
{
    std::unique_lock<std::mutex> lk(this->_mtx);

    if (this->_queue.size() >= this->_queue_size)
    {
        // Queue full
        this->_cv_full.wait(lk, [this]() { return this->_queue.size() < this->_queue_size; });
    }

    this->_queue.push_back(ch);

    lk.unlock();
    this->_cv_empty.notify_all();
}


void HillClimberPool::waitToFinish ()
{
    std::unique_lock<std::mutex> lk(this->_mtx);

    if (!this->_queue.empty())
    {
        // Wait for the queue to get become empty
        this->_cv_full.wait(lk, [this]() { return this->_queue.empty(); });
    }

    lk.unlock();
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //

void HillClimberPool::_launch ()
{
    // In the begining the shut_down is true because the workers are not runing
    assert(this->_shut_down);

    this->_shut_down = false;

    // Determine the number of cores on this machine
    unsigned int num_cores = std::thread::hardware_concurrency();
    std::cout << "-- This machine has " << num_cores << " concurent threads" << std::endl;
    for (int i = 0; i < num_cores; ++i)
    {
        this->_worker_pool.emplace_back(&HillClimberPool::_workerThread, this);
    }
}


void HillClimberPool::_workerThread ()
{
    int worker_id = this->_num_running_workers++;
    syncPrint("-- HILL CLIMBER WORKER [" + std::to_string(worker_id) + "] starting");

    // Create a new HillClimber optimizer
    HillClimber hc(false);

    while (!this->_shut_down)
    {
        std::unique_lock<std::mutex> lk(this->_mtx);

        if (this->_queue.empty())
        {
            // Queue empty
            this->_cv_empty.wait(lk, [this]() { return !this->_queue.empty() || this->_shut_down; });

            if (this->_shut_down) break;  // We do NOT wait for the queue to get empty
        }

        // Get a chromozome from the queue
        std::shared_ptr<Chromozome> chromozome = this->_queue.front();
        this->_queue.pop_front();

        lk.unlock();
        this->_cv_full.notify_all();


        // Run Hill Climber on this chromozome and replace it with the optimized one
        chromozome->update(hc.run(chromozome));
    }

    this->_num_running_workers--;
    syncPrint("-- HILL CLIMBER WORKER [" + std::to_string(worker_id) + "] shutting down");
}


}

