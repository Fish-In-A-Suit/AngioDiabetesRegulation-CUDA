#pragma once

#include <iostream>;
#include <bitset>
#include <vector>

class PermutationUtils
{
public:
    ~PermutationUtils();
    static std::vector<int> generatePermutations(int, std::vector<int>);
    static void generatePermutationsRecursively(std::vector<int>, int);
    static void printPermutations(std::vector<int>);

private:
    PermutationUtils();
    static std::vector<int> bitShiftPrimary(int, int, std::vector<int>);
    static std::vector<int> bitShiftPrimary(std::vector<int>, int, std::vector<int>);
};

