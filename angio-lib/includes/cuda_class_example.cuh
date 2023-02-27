// cuda_class_example.cuh

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

class CudaExampleClass {
public:
	CudaExampleClass(int size);
	~CudaExampleClass();
	void test_function();
	void test_char_copy_function(const char**, const char**, int);
	void test_char_copy_function_v2(char* result_char_array, char* input_char_array, int size);
	void test_char_copy_function_v3(char(*)[10], char(*)[10], int);
private:
	int* d_num;
};