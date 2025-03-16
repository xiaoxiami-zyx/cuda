#include "kernel.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <vector>
int main()
{
    const int N               = 32 * 1024 * 1024;
    int       threadsPerBlock = 256;
    int       block_num       = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> input(N);
    for (int i = 0; i < N; ++i)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }
    std::vector<float> res(block_num, 0.0);
    float              result = 0.0;
    for (int i = 0; i < block_num; ++i)
    {
        for (int j = 0; j < threadsPerBlock; ++j)
        {
            res[i] += input[i * threadsPerBlock + j];
        }
        result += res[i];
    }

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    cudaMemcpy(d_input, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int sharedMemSize = threadsPerBlock * sizeof(float);
    reduce6<<<block_num, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);

    std::vector<float> output(block_num);
    cudaMemcpy(output.data(), d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    float result_gpu = 0.0;
    for (int i = 0; i < block_num; ++i)
    {
        if (std::abs(output[i] - res[i]) > 1e-5)
        {
            printf("Error: output[%d] = %f, expected %f\n", i, output[i], res[i]);
            // return -1;
        }
        result_gpu += output[i];
    }
    printf("Result from cpu: %f\n Result from gpu: %f\n", result, result_gpu);

    cudaFree(d_input);
    cudaFree(d_output);

    thrust::device_vector<float> d_input2(N);
    cudaMemcpy(
        thrust::raw_pointer_cast(d_input2.data()),
        input.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice);
    // reduce
    float result_thrust =
        thrust::reduce(d_input2.begin(), d_input2.end(), 0.0f, thrust::plus<float>());
    printf("Result from thrust: %f\n", result_thrust);
    return 0;
}