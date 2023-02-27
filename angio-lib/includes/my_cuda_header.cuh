#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void my_kernel(int* d_c, int* d_a, int* d_b);