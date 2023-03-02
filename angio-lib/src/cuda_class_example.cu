// cuda_class_example.cu

#include "cuda_class_example.cuh"

/*
 * A device (GPU) implementation of the std::strlen function to find the number of characters inside a const char* cstr.
 */
__device__ int my_strlen_dev(const char* cstr) {
    int len = 0;
    while (cstr[len] != '\0') {
        len++;
    }
    return len;
}

__global__ void exampleKernel(int* num) {
    *num += 1;
}

/*
 * Call with charCopyKernel<<<N,1>>> to execute the addKernel_parallel N times
 * in parallel. By using blockIdx.x to index the array, each block handles a different element of the
 * array. Schematic:
 *   - BLOCK 0: c[0] = a[0] + b[0]
 *   - BLOCK 1: c[1] = a[1] + b[1]
 *   - ...
 * In the above calling example, each block executes a single thread.
 */
__global__ void charCopyKernel(const char** device_result_array, const char** device_char_array) {
    printf("Running parallel thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    device_result_array[blockIdx.x] = device_char_array[blockIdx.x];
    // my_strlen_dev(device_result_array[blockIdx.x]); // this produces an error, unable to read memory

    // attemt - code functionality of my_strlen_dev inside this kernel
    // TODO: THIS ALSO FAILS; I GUESS THAT EVEN device_char_array[0][0] = const char* FIRST_CSTR[0] fails.
    // int len = 0;
    // while (device_char_array[blockIdx.x][len] != '\0') {
    //    len++;
    //}
    // printf("  - currents cstr len: %d\n", len);

    // this works
    const char* test_cstr = "attc";
    printf(test_cstr); printf("\n");

    int len = 0;
    while (test_cstr[len] != '\0') { // this works too
        len++;
    }
    printf("  - current cstr len: %d\n", len);
}

// Tests const char* assignment in a CUDA kernel.
__global__ void charCopyKernelv2(const char** device_result_array, const char** device_char_array) {
    printf("Running parallel thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    const char* current_cstr = device_result_array[blockIdx.x]; // this works
    printf(current_cstr); // nothing is displayed
    // printf("  - current strlen is: %d\n", my_strlen_dev(current_cstr)); // MEMORY ACCESS VIOLATION ERROR because of string literals
    device_result_array[blockIdx.x] = current_cstr;

    // test char s1[]
    char s1[] = "Hello";
    char cs[3][10] = { "Hello", "World", "Cstr" };
    printf("[DEBUG]: test s1 'Hello' length = %d\n", my_strlen_dev(s1));
    printf("[DEBUG]: test double char array: index = %d, result = %s\n", blockIdx.x, cs[blockIdx.x]);
}

/*
*Call with charCopyKernelv3 << <N, 1 >> > to execute the addKernel_parallel N times
* in parallel.By using blockIdx.x to index the array, each block handles a different element of the
* array.Schematic:
*- BLOCK 0 : c[0] = a[0] + b[0]
* -BLOCK 1 : c[1] = a[1] + b[1]
* -...
* In the above calling example, each block executes a single thread.
*/
__global__ void charCopyKernelv3(char(*device_result_char_array)[10], char(*device_input_char_array)[10]) {
    // device_result_char_array[blockIdx.x] = device_input_char_array[blockIdx.x]; THIS DOESNT WORK

    // Explanation: device_result_char_array[blockIdx.x] is the pointer to the first char array (eg. "Hello", wheras other
    // char arrays could be "World", "from", "..."). You cannot assign a pointer to a pointer, hence the "expression must
    // be a modifiable lvalue" error. You need to copy the elements of the subarray element-by-element, hence the for loop.
    
    
    for (int i = 0; i < 10; i++) {
        device_result_char_array[blockIdx.x][i] = device_input_char_array[blockIdx.x][i];
    }
}

/*
 * Call with charCompareKernel<<<N,1>>> to execute the addKernel_parallel N times
 * in parallel. By using blockIdx.x to index the array, each block handles a different element of the
 * array. Schematic:
 *   - BLOCK 0: c[0] = a[0] + b[0]
 *   - BLOCK 1: c[1] = a[1] + b[1]
 *   - ...
 * In the above calling example, each block executes a single thread.
 */
__global__ void charCompareKernel(int* compare_results, const char** device_char_array1, const char** device_char_array2) {
    printf("Running parallel thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    //const char* current_cstr1 = device_char_array1[blockIdx.x];
    //const char* current_cstr2 = device_char_array2[blockIdx.x];
}

//__host__ __device__
CudaExampleClass::CudaExampleClass(int num) {
    cudaMalloc(&d_num, sizeof(int));
    cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice);
}

CudaExampleClass::~CudaExampleClass() {
    cudaFree(d_num);
}

//__host__ __device__
void CudaExampleClass::test_function() {
    exampleKernel<<<1,1>>>(d_num);
    int num;
    cudaMemcpy(&num, d_num, sizeof(int), cudaMemcpyDeviceToHost);
    printf("num: %d\n", num);
}

void CudaExampleClass::test_char_copy_function(const char** result_char_array, const char** input_char_array, int size) {
    // declare GPU copies
    const char** device_char_array;
    const char** device_result_array;

    // set the primary GPU
    cudaSetDevice(0);

    // allocate GPU memory
    cudaMalloc((void**)&device_char_array, size * sizeof(const char*));
    cudaMalloc((void**)&device_result_array, size * sizeof(const char*));

    // copy data from host to device
    cudaMemcpy(device_char_array, input_char_array, size * sizeof(const char*), cudaMemcpyHostToDevice);

    // ERROR: Expression must be a modifiable lvalue here
    // device_char_array[1][2] = '4';
    // *(*(device_char_array + 1) + 2) = '4';

    // run the GPU kernel
    // charCopyKernel<<<size, 1 >>>(device_result_array, device_char_array); // this works
    charCopyKernelv2<<<size, 1 >>>(device_result_array, device_char_array); // this also works

    // copy back to host
    cudaMemcpy(result_char_array, device_char_array, size * sizeof(const char*), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(device_char_array);
    cudaFree(device_result_array);
}

void CudaExampleClass::test_char_copy_function_v3(char(*result_char_array)[20000] , char(*input_char_array)[20000] , int size) {
    // TODO: PROCESS INPUT CHAR[][] FROM MAIN.CU
    char(*device_input_char_array)[10];
    char(*device_result_char_array)[10];

    cudaSetDevice(0);

    cudaMalloc((void**)&device_input_char_array, size * 10 * sizeof(char));
    cudaMalloc((void**)&device_result_char_array, size * 10 * sizeof(char));

    cudaMemcpy(device_input_char_array, input_char_array, size * 10 * sizeof(char), cudaMemcpyHostToDevice);

    charCopyKernelv3<<<size,1>>>(device_result_char_array, device_input_char_array);

    cudaMemcpy(result_char_array, device_result_char_array, size * 10 * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(device_input_char_array);
    cudaFree(device_result_char_array);    
}