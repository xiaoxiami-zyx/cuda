#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

__global__ void fill_kernal(int* a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = i;
}

void fill(int* a, int n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    fill_kernal<<<numBlocks, blockSize>>>(a, n);
}

int main()
{
    constexpr int              N = 10000;
    thrust::device_vector<int> a(N);
    fill(thrust::raw_pointer_cast(a.data()), N);

    int sum = thrust::reduce(a.begin(), a.end(), 0);

    int sum_check = (N - 1) * N / 2;

    if (sum == sum_check)
        std::cout << "CUDA sum reduction is correct" << std::endl;
    else
        std::cout << "CUDA sum reduction is incorrect" << std::endl;

    return 0;
}