
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "StringUtils.h"
#include "my_cuda_header.cuh"
#include "cuda_class_example.cuh"
#include "CUDASequenceComparator.cuh"

#include <stdio.h>
#include <iostream>
#include <vector>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void addWithCudaSimple(int* c, const int* a, const int* b, unsigned int size);
void addWithCudaParallel(int* c, const int* a, const int* b, unsigned int size); 
// void cudaCompareSequences(int* matchStrengths, std::vector<std::string>& mRNAs, std::vector<std::string>& miRNAs, unsigned int strArrSize);
char** convertArray(std::vector<std::string>&, const int);
int my_strlen(const char*); 


__global__ void addKernel(int *c, const int *a, const int *b)
{
    printf("Running thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*
 * Call with addKernel_parallel<<<N,1>>> to execute the addKernel_parallel N times
 * in parallel. By using blockIdx.x to index the array, each block handles a different element of the
 * array. Schematic:
 *   - BLOCK 0: c[0] = a[0] + b[0]
 *   - BLOCK 1: c[1] = a[1] + b[1]
 *   - ...
 * In the above calling example, each block executes a single thread.
 */
__global__ void addKernel_parallel(int* c, const int* a, const int* b) {
    printf("Running parallel thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main()
{
    const int char_array_size = 3;
    std::vector<std::string> strings1 = { "Hello_\0", "World\0", "123\0" };

    // BEGINNER EXAMPLES
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    CudaExampleClass cudaExampleClass(5);
    cudaExampleClass.test_function();
    // --- *** ---
   
    // IMPLEMENTATION OF A FUNCTION TO ACCEPT ANY SIZE OF A CHAR ARRAY (attempt 2)
    char c_strings6[char_array_size][Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH];
    StringUtils::init_Cstrings_array(c_strings6, char_array_size);
    StringUtils::convert_strings_to_Cstrings(c_strings6, strings1, char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
    // StringUtils::convert_strings_to_Cstrings_ptr(c_strings6, strings1, char_array_size, MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
    std::vector<std::string> result_strings6; // convert back into strings
    StringUtils::convert_Cstrings_to_strings(result_strings6, c_strings6, char_array_size, Constants::MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
    // StringUtils::convert_Cstrings_to_strings_ptr(result_strings6, c_strings6, char_array_size, MAX_CHAR_ARRAY_SEQUENCE_LENGTH);
    StringUtils::print_vector(result_strings6, "c_strings6 result:");
    

    // EXAMPLE: COPYING CHAR ARRAYS FROM CPU TO GPU AND BACK
    // disadvantage of using double pointers: we cannot modify the values on the gpu, only read them
    char strings2[3][10] = { "Hello1", "World1", "1234" };
    char result_strings2[3][10] = {};

    std::vector<char(*)[10]> vec; // char(*)[10] is a pointer to a character array !!!
    for (int i = 0; i < char_array_size; i++) {
        vec.push_back(&strings2[i]);
    }

    const char** cstrings = new const char* [strings1.size()]; // each element is a pointer to a C-style string
    for (int i = 0; i < strings1.size(); i++) {
        cstrings[i] = strings1[i].c_str();
    }
    std::cout << "Displaying cstrings" << std::endl;
    for (int i = 0; i < strings1.size(); i++) {
        std::cout << cstrings[i] << std::endl;
    }

    const char** result_cstrings = new const char* [strings1.size()];
    cudaExampleClass.test_char_copy_function(result_cstrings, cstrings, strings1.size());
    cudaDeviceSynchronize();
    std::cout << "Printing c-style strings sent to and copied back from the gpu:" << std::endl;
    for (int i = 0; i < strings1.size(); i++) {
        std::cout << "  - " << result_cstrings[i] << std::endl;
    }
    // --- *** ---

    // CUDA SEQUENCE COMPARATOR
    printf("Starting CUDA Sequence Comparator.\n");
    CUDASequenceComparator cudaSequenceComparator("src_data_files/mirbase_miRNA_hsa-only_one.txt", "src_data_files/product_mRNAs_cpp_one.txt");
    // cudaSequenceComparator.compare_sequences();
    // cudaSequenceComparator.compare_sequences_debug();
    cudaSequenceComparator.compare_sequences_v2(true); // this is the best so far
    cudaDeviceSynchronize();
    cudaSequenceComparator.save_sequence_comparison_results("src_data_files/sequence_comparison_results.txt");


    // ANOTHER STARTER EXAMPLE
    // warning: this must be at the end after all cuda calls, since it somehow messes up if it is called before other cuda calls.
    addWithCudaSimple(c, a, b, arraySize);
    // addWithCudaParallel(c, a, b, arraySize);
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    // --- *** ---

    return 0;
}

// Function to add vectors a and b into c, without the error checks
void addWithCudaSimple(int* c, const int* a, const int* b, unsigned int size) {
    // declare device (GPU) copies
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // set the primary GPU (change this on multi-gpu devices)
    cudaSetDevice(0);

    // allocate GPU memory
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // copy data from host to device
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // run the gpu kernel
    addKernel<<<1,size>>>(dev_c, dev_a, dev_b);

    // test the gpu kernel from .cu / .cuh separation
    my_kernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // copy result back to host
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return;
}

void addWithCudaParallel(int* c, const int* a, const int* b, unsigned int size) {
    // declare device (GPU) copies
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // set the primary GPU (change this on multi-gpu devices)
    cudaSetDevice(0);

    // allocate GPU memory
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // copy data from host to device
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // run the gpu kernel
    addKernel_parallel<<<size, 1>>>(dev_c, dev_a, dev_b);

    // copy result back to host
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return;
}

// Helper function for using CUDA to add vectors in parallel.
// This function also has error-checking functionality at each step of the way.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    // device (GPU) copies of a, b and c input params
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1,size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

/*
 * Return type is char** (double pointer) as we are returning an array of char* ponters.
 * 
 */
char** convertArray(std::vector<std::string>& string_array, const int arraySize) {
    // create new array to hold the char arrays
    char** result = new char*[arraySize];

    // convert each element
    for (int i = 0; i < arraySize; i++) {
        const char* str = string_array[i].c_str();
        result[i] = new char[strlen(str) + 1];
        strcpy(result[i], str);
    }

    // return the array of char arrays
    return result;
}

/*
 * A test implementation of the std::strlen function to be used in the CUDA kernel.
 */
int my_strlen(const char* cstr) {
    int len = 0;
    while (cstr[len] != '\0') { // !! WARNING !!: using "\0" produces an error as it is regarded as const char*, whereas using '\0' is regarded as char
        len++;
    }
    return len;
}


