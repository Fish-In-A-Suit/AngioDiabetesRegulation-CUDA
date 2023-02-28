#pragma once

#include <chrono>
#include <cmath>
#include "StringUtils.h"
#include "Constants.h"
#include <iostream>
#include <sstream>

class HighResolutionTimeManagerV2
{
    std::chrono::high_resolution_clock::time_point startTime;
public:
    HighResolutionTimeManagerV2();
    ~HighResolutionTimeManagerV2();
    void setStartTime();
    std::string getElapsedTime(Constants::TimeUnits, bool);
private:
};

