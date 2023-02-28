#pragma once

#include <string>
#include <filesystem>
#include <iostream>

class FileUtils
{
public:
    FileUtils(int);
    ~FileUtils();
    static std::pair<std::string, const char*> getAbsoluteFilepath(std::string);
    static std::string getProjectRootPath();
    static void setProjectRootPath(std::string);
private:
    inline static std::string projectRootPath; //check 'inline' here: https://stackoverflow.com/questions/9110487/undefined-reference-to-a-static-member
};

