/**
 * @file stream_order_memory.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示基于流顺序内存分配API(cudaMallocAsync/cudaFreeAsync)的基本用法
 * @version 0.1
 * @date 2026-03-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "error.cu"
#include <cuda_runtime_api.h>

__global__ void kernel(int* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = idx;
}

void test_stream_order_memory()
{
    void*  ptr  = nullptr;
    int    n    = 128;
    size_t size = static_cast<size_t>(n) * sizeof(int);

    CHECK_CUDA(cudaMallocAsync(&ptr, size, cudaStreamPerThread));

    // 在同一流上使用这块异步分配的内存。
    kernel<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(static_cast<int*>(ptr), n);
    CHECK_CUDA(cudaGetLastError());

    // 异步释放：释放操作会按流顺序在内核之后执行。
    CHECK_CUDA(cudaFreeAsync(ptr, cudaStreamPerThread));

    // 仅用于示例收尾，确保当前流上的操作执行完成。
    CHECK_CUDA(cudaStreamSynchronize(cudaStreamPerThread));
}

void test_stream_order_memory_free()
{
    void*        ptr  = nullptr;
    int          n    = 128;
    size_t       size = static_cast<size_t>(n) * sizeof(int);
    cudaStream_t stream1, stream2, stream3;
    cudaEvent_t  event1, event2;

    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventCreate(&event2));

    cudaMallocAsync(&ptr, size, stream1);
    cudaEventRecord(event1, stream1);

    // stream2必须等待分配完成后才能访问该指针。
    cudaStreamWaitEvent(stream2, event1);
    kernel<<<(n + 255) / 256, 256, 0, stream2>>>(static_cast<int*>(ptr), n);
    CHECK_CUDA(cudaGetLastError());
    cudaEventRecord(event2, stream2);

    // stream3必须等待stream2完成访问后才能释放内存。
    cudaStreamWaitEvent(stream3, event2);
    cudaFreeAsync(ptr, stream3);

    CHECK_CUDA(cudaStreamSynchronize(stream3));
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaEventDestroy(event2));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
}