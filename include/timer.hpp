#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <iostream>
#include <string>
#include <time.h>
#include <sys/time.h>

namespace util {

  using tick_t = long double;

  class timer {
    timespec startTime;
    timespec stopTime;
    tick_t elapsedTime { 0 };

  public:

    inline void start() {
      clock_gettime(CLOCK_MONOTONIC, &startTime);
    }

    inline void stop() {
      clock_gettime(CLOCK_MONOTONIC, &stopTime);
      elapsedTime = static_cast <tick_t> (stopTime.tv_sec - startTime.tv_sec) + static_cast <tick_t>(stopTime.tv_nsec - startTime.tv_nsec) / 1.0e9;
    }

    inline tick_t elapsed() {
      return (elapsedTime);
    }
  };

  class block_timer {
    std::string tag;
    timer t;

  public:

    inline block_timer (std::string _tag)
      : tag { _tag } {
      t.start();
    }

    inline ~block_timer() {
      t.stop();
      std::cout << tag << ": " << t.elapsed() << std::endl;
    }
  };
}

#endif
