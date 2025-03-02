#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

int main()
{
    const int                  N = 10000;
    thrust::device_vector<int> a(N);
    thrust::sequence(a.begin(), a.end(), 0);

    int sum = thrust::reduce(a.begin(), a.end(), 0);

    int sum_check = (N - 1) * N / 2;

    if (sum == sum_check)
        std::cout << "CUDA sum reduction is correct" << std::endl;
    else
        std::cout << "CUDA sum reduction is incorrect" << std::endl;

    return 0;
}