//
// Libor Novak
// 11/11/2016
//

#ifndef NEWCHROMOZOMEPOOL_H
#define NEWCHROMOZOMEPOOL_H

#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "components/Chromozome.h"


namespace eic {


class NewChromozomePool
{
public:

    NewChromozomePool (const std::shared_ptr<const Target> &target, int pool_size);

    /**
     * @brief Starts the generator of new chromozomes
     */
    void launch ();

    /**
     * @brief Shuts down the generator of new chromozomes
     */
    void shutDown ();

    std::shared_ptr<Chromozome> getNewChromozome ();


private:

    /**
     * @brief Worker that will be filling up the queue
     */
    void _workerThread ();
    

    // -------------------------------------  PRIVATE MEMBERS  ------------------------------------- //
    const std::shared_ptr<const Target> _target;
    // Max number of generated chromozomes in the queue
    int _pool_size;
    // Thread that will be generating new chromozomes
    std::thread _worker;
    // Queue of new chromozomes
    std::deque<std::shared_ptr<Chromozome>> _queue;
    // Queue access logic
    std::mutex _mtx;
    std::condition_variable _cv_full;
    std::condition_variable _cv_empty;
    std::atomic<bool> _shut_down;

};


}


#endif // NEWCHROMOZOMEPOOL_H
