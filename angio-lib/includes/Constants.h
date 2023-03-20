#pragma once
class Constants
{
public:
    // reasons for implementing enum in .h: https://stackoverflow.com/questions/1284529/enums-can-they-do-in-h-or-must-stay-in-cpp
    enum TimeUnits
    {
        NANOSECONDS,
        MICROSECONDS,
        MILLISECONDS,
        SECONDS,
        MINUTES,
        HOURS
    };

    enum LogLevels {
        DEBUG,
        INFO
    };

    static const int MAX_CHAR_ARRAY_SEQUENCE_LENGTH = 15000;

    static const int MRNA_UNIPROT_ID_LENGTH = 17; // 16 chars for uniprot id and additional char for \0
    static const int MIRNA_MI_ID_LENGTH = 10; // 9 chars for MI id and additional char for \0
};

