#include <stdio.h>

int main()
{
    int iDevice = 0;
    cudaGetDeviceCount(&iDevice);
    printf("Number of CUDA devices: %d\n", iDevice);

    cudaSetDevice(0);
}