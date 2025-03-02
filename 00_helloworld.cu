#include <stdio.h>

__global__ void hello_from_gpu()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    printf("Hello from GPU!%d\n", idx);
}

int main()
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}