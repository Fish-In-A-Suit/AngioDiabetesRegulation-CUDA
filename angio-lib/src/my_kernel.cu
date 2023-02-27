// my_kernel.cu - device code
#include "my_cuda_header.cuh"

/*
 * This is called in the addWithCudaSimple method of main.cu, make sure to include
 * #include "my_cuda_header.cuh"
 */
__global__ void my_kernel(int* d_c, int* d_a, int* d_b) {
    printf("Running thread [%d,%d]\n", threadIdx.x, blockIdx.x);
    int i = threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}