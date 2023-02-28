#pragma once

// todo: how to use 3rd party library in nested folders ?
//#include "rapidjson/document.h"
#include <rapidjson/document.h> // rapidjson
#include <rapidjson/filereadstream.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <iostream>

class JsonObject
{
public:
    JsonObject(std::string, int, bool);
    ~JsonObject();
    void setJson(std::string, int);
    // rapidjson::Document getJsonDoc(); // cannot return due to it being prohibited by document.h
    const char* getValue(std::string);
    std::string toString(bool);

    bool getAssertionStatus();
    void setAssertionStatus(bool);
private:
    std::string filepath;
    rapidjson::Document jsonDoc;
    bool checkAssertions;
};

