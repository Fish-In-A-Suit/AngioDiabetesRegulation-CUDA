// StringUtils.cpp

#include "StringUtils.h"

// class implementation belongs to the .cpp file

// you can implement constructor params like so:
// StringUtils::StringUtils(int test) {
//     testInt = test;
// }

// empy constructor implementation
// note: no ';' after class brackets!
StringUtils::StringUtils() {

}

// destructor implementation
StringUtils::~StringUtils() {
    std::cout << "StringUtils destructor called" << std::endl;
}

std::string StringUtils::to_string(double value, int precision) {
    /*
    std::ostringstream streamObj;              // create an output string stream
    streamObj << std::fixed;                   // set fixed-point notation
    streamObj << std::setprecision(precision); // set precision
    streamObj << value;                        // add number to stream
    return streamObj.str();                    // get string from output string stream
    */
    return "THIS FUNCTIONALITY IS DEPRECATED.";
}

/*
 * Splits 'str' string based on delimiter and populates the remaining split elements inside the supplied
 * std::vector<std::string>
 *
 * @param str: The string to split
 * @param delimiter: The delimiter string where the splits occur
 * @param line_elements: an empty std::vector<std::string> to populate
 *
 * @return std::vector<std::string> of remaining elements after split
 */
void StringUtils::split(std::string& str, std::string delimiter, std::vector<std::string>& line_elements) {
    size_t pos = 0;
    std::string token;

    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        line_elements.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    return;
}

bool StringUtils::contains(std::string& str, std::string& substr) {
    // std::cout << "str = " << str << ", substr = " << substr << ", str.find(substr) = " << str.find(substr) << std::endl;
    if (str.find(substr) != std::string::npos) {
        // std::cout << "true" << std::endl;
        return true; // found
    }
    else {
        // std::cout << "false" << std::endl;
        return false; // not found
    }
}

bool StringUtils::contains(std::string& str, const char* substr) {
    std::string substr_str = substr; // std::string has a constructor from const char *, so this is legal (https://stackoverflow.com/questions/24127946/converting-a-const-char-to-stdstring)
    return contains(str, substr_str);
}

/*
 * Prints the std::vector<std::string> string elements line by line to std::cout. Pass-by-reference is used (&vecString), since the contents of
 * std::vector are only read.
 *
 * @param vecString: A vector of strings whose elements to print to std::cout
 * @param firstMessage: A message that should be displayed before the loop prints out all of the contents
 * @param shouldIndent: If the contents of the loop should be indented. Default is true.
 * @param indentSeparator: Which indent separator to use (default is "\t")
 */
void StringUtils::print_vector(std::vector<std::string>& vecString, std::string firstMessage, bool shouldIndent, std::string indentSeparator) {
    if (firstMessage != "") {
        std::cout << firstMessage << std::endl;
    }

    if (shouldIndent) {
        for (std::string str : vecString) {
            std::cout << indentSeparator << str << std::endl;
        }
    }
    else {
        for (std::string str : vecString) {
            std::cout << str << std::endl;
        }
    }
}

/*
 * Converts a vector of strings into the supplied pointer to a 2D char array. For example, {"Hello", "World", "T"} is converted to the following
 * single-character arrays (numbers like 01 correspond to indices [i][j]):
 * 
 * 00 01 02 03 04 05
 * H  e  l  l  o  \0
 * 
 * 10 11 12 13 14 15
 * W  o  r  l  d  \0
 * 
 * 20 21 22 23 24 25
 * T  \0 \0 \0 \0 \0
 * 
 * 'row_count' represents the amount of strings you wish to convert from 'strings' (in our example: row_count = 3)
 * 'col_count' represents the maximum size of the character array, that is the maximum amount of characters an input string can contain. 'col_count'
 *  should be set to MAX_CHAR_ARRAY_SEQUENCE_LENGTH.
 * 
 * In particular, C++ requires us to specify the second dimension (column size) of an array at compile time, so the compiler knows how much memory
 * it must allocate for the array. So, whenever you are creating char arrays in code, use the MAX_CHAR_ARRAY_SEQUENCE_LENGTH as the 2nd dimension.
 * 
 * To create an empty 2D array of chars in code, use the following snippet:
 * 
 *      your_char_array_size = 100;
 *      char c_strings[your_char_array_size][Constants::MAX_CHAR_SEQUENCE_LENGTH];
 *      StringUtils::init_Cstrings_array(c_strings, your_char_array_size);
 * 
 *      // example of using pointer logic to access each element
 *      char(*c_strings_ptr)[char_element_size] = c_strings;
 *      std::cout << *(*(c_strings_ptr + i)+j) << std::endl; // access i'th row and j'th column of c_strings_ptr (which points to address held by [0][0]'th element = first element)
 * 
 * This will populate each [i][j] char element inside the 2D array with a \0 by default. TODO: create a helper function 'create_empty_cstring_array_2D' which returns a char[][MAX_CHAR_ARRAY_SEQUENCE_LENGTH]
 * 
 * Then, when you have a vector of strings and want to copy them into the created char array, use the following:
 * 
 *      std::vector<std::string> vec_strings = {...};
 *      StringUtils::convert_strings_to_Cstrings_ptr(c_strings, vec_strings, your_char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
 *      // or
 *      StringUtils::convert_strings_to_Cstrings(c_strings, vec_strings, your_char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
 * 
 * The usage between convert_strings_to_Cstrings and the "ptr" variant (convert_strings_to_Cstrings_ptr) is none, both functions produce the same output.
 * The only difference is the internal processing - the "ptr" function variant uses pointer logic, whereas the non-ptr function avoids pointer logic.
 * 
 * @param input_char_array_ptr: an empty char[][] array, created as described above
 * @param strings: a std::vector array of std::strings
 * @param row_count: corresponds to strings.size() and represents the first-dimension of the 2D array (amount of rows)
 * @param col_count: corresponds to the maximum string length than can be stored (strings must not surpass the length set by Constants::MAX_CHAR_ARAY_SEQUENCE_LENGTH)
 * 
 * @return: none, but populates input_char_array_ptr (pointing to the 2d char array) with strings.
 * 
 * Note: the input char pointer is passed by reference! Normally, the pointers are passed by values, therefore local copies are made. If you want to modify the contents
 * of the existing pointer, you have to pass it by reference using '&'.
 */
void StringUtils::convert_strings_to_Cstrings_ptr(char(*&input_char_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], std::vector<std::string>& strings, int row_count, int col_count) {
    int strings_size = strings.size();
    // todo: check if size of input_char_array (the first dimensiuon) = row_count and strings_size match
    for (int i = 0; i < strings_size; i++) {
        std::string current_string = strings[i];
        int string_length = current_string.length();
        // todo: check if the size of string_length is less than the second dimension of input_char_array (represented by col_count)

        for (int j = 0; j < string_length; j++) {
            char c = current_string.at(j);
            *(*(input_char_array_ptr + i) + j) = c; // access element at row i, column j
        }

    }
}

/*
 * After using StringUtils::convert_strings_to_Cstrings or StringUtils::convert_strings_to_Cstrings_ptr to populate a 2D char array, it is useful
 * to be able to convert it back to a std::vector<std::string>, which this function does.
 * 
 * Note: std::vector is usually passed by value. Here, we use pass by reference (&) in order to make changes to the input std::vector arrays (without making copies).
 * Note: there is no difference between the "ptr" and the non-ptr variant of this function except in internal logic.
 * 
 * @param dst_strings: an empty std::vector<std::string>
 * @param input_char_array_ptr: a pointer to the char 2D array
 * @param row_count: corresponds to strings.size() and represents the first-dimension of the 2D array (amount of rows)
 * @param col_count: corresponds to the maximum string length than can be stored (strings must not surpass the length set by Constants::MAX_CHAR_ARAY_SEQUENCE_LENGTH)
 * 
 */
void StringUtils::convert_Cstrings_to_strings_ptr(std::vector<std::string>& dst_strings, char(*input_char_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int row_count, int col_count) {
    for (int i = 0; i < row_count; i++) {
        std::string row_str(*(input_char_array_ptr + i), col_count);
        dst_strings.push_back(row_str);
    }
}

/*
 * Converts a vector of strings into the supplied 2D char array. For example, {"Hello", "World", "T"} is converted to the following
 * single-character arrays (numbers like 01 correspond to indices [i][j]):
 *
 * 00 01 02 03 04 05
 * H  e  l  l  o  \0
 *
 * 10 11 12 13 14 15
 * W  o  r  l  d  \0
 *
 * 20 21 22 23 24 25
 * T  \0 \0 \0 \0 \0
 *
 * 'row_count' represents the amount of strings you wish to convert from 'strings' (in our example: row_count = 3)
 * 'col_count' represents the maximum size of the character array, that is the maximum amount of characters an input string can contain. 'col_count'
 *  should be set to MAX_CHAR_ARRAY_SEQUENCE_LENGTH.
 *
 * In particular, C++ requires us to specify the second dimension (column size) of an array at compile time, so the compiler knows how much memory
 * it must allocate for the array. So, whenever you are creating char arrays in code, use the MAX_CHAR_ARRAY_SEQUENCE_LENGTH as the 2nd dimension.
 *
 * To create an empty 2D array of chars in code, use the following snippet:
 *
 *      your_char_array_size = 100;
 *      char c_strings[your_char_array_size][Constants::MAX_CHAR_SEQUENCE_LENGTH];
 *      StringUtils::init_Cstrings_array(c_strings, your_char_array_size);
 *
 * This will populate each [i][j] char element inside the 2D array with a \0 by default. TODO: create a helper function 'create_empty_cstring_array_2D' which returns a char[][MAX_CHAR_ARRAY_SEQUENCE_LENGTH]
 *
 * Then, when you have a vector of strings and want to copy them into the created char array, use the following:
 *
 *      std::vector<std::string> vec_strings = {...};
 *      StringUtils::convert_strings_to_Cstrings_ptr(c_strings, vec_strings, your_char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
 *      // or
 *      StringUtils::convert_strings_to_Cstrings(c_strings, vec_strings, your_char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
 *
 * The usage between convert_strings_to_Cstrings and the "ptr" variant (convert_strings_to_Cstrings_ptr) is none, both functions produce the same output.
 * The only difference is the internal processing - the "ptr" function variant uses pointer logic, whereas the non-ptr function avoids pointer logic.
 *
 * @param input_char_array: an empty char[][] array, created as described above
 * @param strings: a std::vector array of std::strings
 * @param row_count: corresponds to strings.size() and represents the first-dimension of the 2D array (amount of rows)
 * @param col_count: corresponds to the maximum string length than can be stored (strings must not surpass the length set by Constants::MAX_CHAR_ARAY_SEQUENCE_LENGTH)
 *
 * @return: none, but populates input_char_array_ptr (pointing to the 2d char array) with strings.
 * 
 * TODO: CHECK IF THIS WORKS NORMALLY USING PASS BY REFERENCE ???
 * ADVICE: USE convert_strings_to_Cstrings_ptr, as they surely work.
 */
void StringUtils::convert_strings_to_Cstrings(char input_char_array[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], std::vector<std::string>& strings, int row_count, int col_count) {
    int strings_size = strings.size();
    // todo: check if size of input_char_array (the first dimensiuon) = row_count and strings_size match
    for (int i = 0; i < strings_size; i++) {
        std::string current_string = strings[i];
        int string_length = current_string.length();
        // todo: check if string_length < col_count
        for (int j = 0; j < string_length; j++) {
            char c = current_string.at(j);
            input_char_array[i][j] = c;
        }
    }
}

/*
 * After using StringUtils::convert_strings_to_Cstrings or StringUtils::convert_strings_to_Cstrings_ptr to populate a 2D char array, it is useful
 * to be able to convert it back to a std::vector<std::string>, which this function does.
 *
 * Note: std::vector is usually passed by value. Here, we use pass by reference (&) in order to make changes to the input std::vector arrays (without making copies).
 * Note: there is no difference between the "ptr" and the non-ptr variant of this function except in internal logic.
 *
 * @param dst_strings: an empty std::vector<std::string>
 * @param input_char_array_ptr: a pointer to the char 2D array
 * @param row_count: corresponds to strings.size() and represents the first-dimension of the 2D array (amount of rows)
 * @param col_count: corresponds to the maximum string length than can be stored (strings must not surpass the length set by Constants::MAX_CHAR_ARAY_SEQUENCE_LENGTH)
 *
 */
void StringUtils::convert_Cstrings_to_strings(std::vector<std::string>& dst_strings, char input_char_array[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int row_count, int col_count) {
    for (int i = 0; i < row_count; i++) {
        std::string row_str(input_char_array[i], col_count);
        dst_strings.push_back(row_str);
    }
}

/*
 * Initialises each element (char) in a char 2D array (Cstrings array) to 'default_char'. 
 * 
 * @param dst_cstrings: an empty, but defined 2D char array
 * @param array_size: the size of the array (equals to row_count)
 * @param default_char: the character that will be used to populate the elements of the 2D array
 * 
 * @return void, but populates each element in dst_cstrings with default_char
 * 
 * Example usage:
 * 
 *      your_char_array_size = 100;
 *      char c_strings[your_char_array_size][Constants::MAX_CHAR_SEQUENCE_LENGTH];
 *      StringUtils::init_Cstrings_array(c_strings, your_char_array_size);
 */
void StringUtils::init_Cstrings_array(char dst_cstrings[][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int array_size, char default_char) {
    for (int i = 0; i < array_size; i++) {
        for (int j = 0; j < Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH; j++) {
            dst_cstrings[i][j] = default_char;
        }
    }
}






