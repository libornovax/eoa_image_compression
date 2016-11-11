#include "NewChromozomePool.h"

#include "algorithms/HillClimber.h"
#include "components/Config.h"


namespace eic {


NewChromozomePool::NewChromozomePool (const std::shared_ptr<const Target> &target, int pool_size)
    : _target(target),
      _pool_size(pool_size),
      _shut_down(false)
{

}


void NewChromozomePool::launch ()
{
    this->_shut_down = false;
    this->_worker = std::thread(&NewChromozomePool::_workerThread, this);
}


void NewChromozomePool::shutDown()
{
    this->_shut_down = true;
    // If the queue is full and the thread is waiting, we need to trigger the shutdown
    this->_cv_full.notify_one();

    this->_worker.join();
}


std::shared_ptr<Chromozome> NewChromozomePool::getNewChromozome ()
{
    assert(!this->_shut_down);
    assert(this->_worker.joinable());  // Is the worker runing?

    std::unique_lock<std::mutex> lk(this->_mtx);

    if (this->_queue.empty())
    {
        // Queue empty
        this->_cv_empty.wait(lk);
    }

    // Get a new chromozome from the queue
    auto new_chromozome = this->_queue.front();
    this->_queue.pop_front();

    lk.unlock();
    this->_cv_full.notify_one();

    return new_chromozome;
}


// ------------------------------------------  PRIVATE METHODS  ------------------------------------------ //


void NewChromozomePool::_workerThread ()
{
    while (!this->_shut_down)
    {
        // Generate a new chromozome
        std::shared_ptr<Chromozome> new_chromozome;
        switch (Config::getParams().classic_ea.chromozome_init)
        {
        case ChromozomeInit::RANDOM:
            {
                new_chromozome = Chromozome::randomChromozome(this->_target);
            }
            break;
        case ChromozomeInit::HILL_CLIMBER:
            {
                HillClimber hc(this->_target);
                std::cout << "HILLCLIMBER" << std::endl;
                new_chromozome = hc.run();
            }
            break;
        default:
            std::cout << "ERROR: Unknown chromozome initialization" << std::endl;
            exit(EXIT_FAILURE);
            break;
        }


        // Put the new chromozome into the queue
        std::unique_lock<std::mutex> lk(this->_mtx);

        if (this->_queue.size() >= this->_pool_size)
        {
            // Queue full
            this->_cv_full.wait(lk);

            if (this->_shut_down) break;
        }

        this->_queue.push_back(new_chromozome);

        lk.unlock();
        this->_cv_empty.notify_one();
    }
}


}

