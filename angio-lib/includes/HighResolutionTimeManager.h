#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include "StringUtils.h"
#include "Constants.h"

class HighResolutionTimeManager
{
    std::chrono::high_resolution_clock::time_point startTime;

public:
    HighResolutionTimeManager();
    ~HighResolutionTimeManager();
    void setStartTime();
    long getElapsedTime(Constants::TimeUnits, bool);
    std::string getElapsedTime(Constants::TimeUnits);

private:
    std::string formatTimeValue(long, Constants::TimeUnits);
    std::string _divideTimeAndReturn(long, long, int, std::string, std::string, int);
};

