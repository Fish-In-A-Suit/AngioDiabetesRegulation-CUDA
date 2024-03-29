#pragma once

// class declaration includes are automatically transferred to.cpp implementation
#include <string>
#include <iomanip>
#include <iostream>
#include <vector>
#include "Constants.h"
// never use 'using namespace <...>' in .h !!!

// class declaration is made in .h file

class StringUtils
{
public:
    // StrUtils(int test); // you can define constructor params like so
    ~StringUtils();
    static std::string to_string(double, int); // TODO declare statics here and not in cpp; use StringUtils::to_string to call.; BUT WHY?
    static void split(std::string&, std::string, std::vector<std::string>&);
    static void print_vector(std::vector<std::string>&, std::string = "", bool = true, std::string = "\t");
    static void print_array(int**, int, std::string = "", bool = true, std::string = "\t");
    static bool contains(std::string&, std::string&);
    static bool contains(std::string&, const char*);

    static void convert_strings_to_Cstrings_ptr(char*&, std::vector<std::string>&, int, int);
    static void convert_Cstrings_to_strings_ptr(std::vector<std::string>& strings, char*&, int, int, char=' ');

    static void convert_strings_to_Cstrings(char[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], std::vector<std::string>&, int, int);
    static void convert_Cstrings_to_strings(std::vector<std::string>&, char[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int, int);

    static void convert_strings_to_Cstrings_ptr(char(*&)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], std::vector<std::string>&, int, int);
    static void convert_Cstrings_to_strings_ptr(std::vector<std::string>&, char(*)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int, int);

    // for debugging miRNAids and mRNA ids
    static void convert_strings_to_Cstrings_ptr(char(*&)[Constants::MIRNA_MI_ID_LENGTH], std::vector<std::string>&, int, int); // each miRNA MI type id has 9 chars
    static void convert_strings_to_Cstrings_ptr(char(*&)[Constants::MRNA_UNIPROT_ID_LENGTH], std::vector<std::string>&, int, int); // each mRNA uniprot type id has 16 chars

    static void init_Cstrings_array(char[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int, char = '\0');

    static void reverse_array(char(*&)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int, int*);
    static void reverse_array(char*&, char*, int, int*, int);

private:
    StringUtils(); // constructor declared private, as StringUtils is meant to provide static string manipulation methods
};

