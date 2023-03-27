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

void StringUtils::print_array(int** array_ptr_reference, int arr_len, std::string firstMessage, bool shouldIndent, std::string indentSeparator) {
    if (firstMessage != "") {
        std::cout << firstMessage << std::endl;
    }
    if (shouldIndent) {
        for (int i = 0; i < arr_len; i++) {
            std::cout << indentSeparator << (*array_ptr_reference)[i] << std::endl;
        }
    }
    else {
        for (int i = 0; i < arr_len; i++) {
            std::cout << (*array_ptr_reference)[i] << std::endl;
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
 *                   col_count should be thought of as the pitch used to store the values in memory !!!
 * 
 * @return: none, but populates input_char_array_ptr (pointing to the 2d char array) with strings.
 * 
 * TODO: make use of row_count and col_count, else delete these two parameters
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

// for miRNAids debugging
void StringUtils::convert_strings_to_Cstrings_ptr(char(*&input_char_array_ptr)[Constants::MIRNA_MI_ID_LENGTH], std::vector<std::string>& strings, int row_count, int col_count) {
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

// for mRNAids debugging
void StringUtils::convert_strings_to_Cstrings_ptr(char(*&input_char_array_ptr)[Constants::MRNA_UNIPROT_ID_LENGTH], std::vector<std::string>& strings, int row_count, int col_count) {
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

// this function works for variable char array lengths, assuming they were allocated using malloc before calling this function!
void StringUtils::convert_strings_to_Cstrings_ptr(char*& input_char_array_ptr, std::vector<std::string>& strings, int row_count, int col_count) {
    // TODO: implement allocation size check, to see if the allocated size of the pointer is greater than 0!
    
    int strings_size = strings.size();
    // todo: check if size of input_char_array (the first dimensiuon) = row_count and strings_size match
    for (int i = 0; i < strings_size; i++) { // loop over each miRNA
        std::string current_string = strings[i];
        int string_length = current_string.length();

        for (int j = 0; j < string_length; j++) { // loop over each char in miRNA
            char c = current_string.at(j);
            // *(*(input_char_array_ptr + i) + j) = c; // access element at row i, column j
            input_char_array_ptr[i * col_count + j] = c;
        }

        // append ' ' from the end of string to the end of col_count
        for (int j = string_length; j < col_count; j++) { // append ' ' (as many spaces as col_count - string_length)
            // input_char_array_ptr[i * col_count + j] = '\0'; // VS-Bug: adding \0 to the array, doesn't display the entire array in Visual Studio Debugger Text-Visualiser, but only the array up to \0.
            input_char_array_ptr[i * col_count + j] = ' '; // adding ' ' is better, since the entire array in the debugger is visible
        }
    }
}

// this function works with linear variable char array lengths, assuming they were allocated using malloc before calling this function!
// each string is 1 row, row_count represents the amount of strings in 'src', 'col_count' is the pitch used
// delimiter is the character used to delimit strings (ie. to populate space between the length of the string and the col_count ie. pitch
void StringUtils::convert_Cstrings_to_strings_ptr(std::vector<std::string>& dst, char*& src, int row_count, int col_count, char delimiter) {
    int string_start_index = -1;
    for (int i = 0; i < row_count; i++) {
        string_start_index = i * col_count; // each string in the array is separated by a pitch of col_count
        for (int j = 0; j < col_count; j++) {
            int string_end_index = i * col_count + j; // calculate the end index
            char current_char = src[string_end_index];
            if ((current_char == delimiter) || (string_end_index == col_count*(i+1)-1)) { // if delimiter is hit, or if max pitch is reached; col_count*(i+1) is the condition for the end of this string, -1 substracted due to c++ array notation starting at 0
                std::string row_str(src + string_start_index, src + string_end_index + 1); // BUGFIX: + 1 is necessary here, otherwise a bug occurs where the output string is 1 character too short
                dst.push_back(row_str);
                break;
            }
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

/* 
 * Reverses the elements of 'source_char_array_ptr' and stores the reversed values in 'destination_char_array_ptr'.
 * 
 * Params:
 *   @param destination_char_array_ptr: a pointer to a 2D char array (that must be pre-initialised)
 *   @param source_char_array_ptr: a pointer to a 2D char array
 *   @param array_size: the size of source_char_array_ptr array
 *   @param source_char_array_lengths: an int array of .lengths() of each string (row) in source_char_array_ptr. This can also be set to Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH.
 * 
 * Note: you need to pre-initialise the destination_char_array_ptr. Example:
 * 
 *      char(*mRNA_cstrings_reversed_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH] = new char[mRNA_sequences.size()][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
 *      StringUtils::reverse_array(mRNA_cstrings_reversed_array_ptr, mRNA_cstrings_array_ptr, mRNA_sequences.size(), mRNA_lengths);
 *	    this->mRNA_sequences_chars_reversed = mRNA_cstrings_reversed_array_ptr;
 * 
 */
void StringUtils::reverse_array(char(*&destination_char_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], char(*source_char_array_ptr)[Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH], int array_size, int* source_char_array_lengths) {
    for (int i = 0; i < array_size; i++) {
        char* current_mRNA_sequence = source_char_array_ptr[i];
        for (int j = 0; j < source_char_array_lengths[i]/2; j++) { // this for loop works down from both ends of the array
            *(*(destination_char_array_ptr + i) + j) = current_mRNA_sequence[source_char_array_lengths[i] - 1 - j];
            *(*(destination_char_array_ptr + i) + source_char_array_lengths[i]-1-j) = current_mRNA_sequence[j];
        }
    }
}

// accepts variable length arrays, with mRNA_pitch meaning how far these values are spaced out in memory
void StringUtils::reverse_array(char*& destination_char_array_ptr, char* source_char_array_ptr, int array_size, int* source_char_array_lengths, int mRNA_pitch) {
    for (int i = 0; i < array_size; i++) {
        int current_mRNA_length = source_char_array_lengths[i];
        for (int j = 0; j < mRNA_pitch; j++) {
            int index = i * mRNA_pitch + j;
            if (j > current_mRNA_length) {
                destination_char_array_ptr[index] = '\0';
            }
            else {
                destination_char_array_ptr[index] = source_char_array_ptr[mRNA_pitch - 1 - index];
            }
        }
    }
}






