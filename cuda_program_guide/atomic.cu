/**
 * @file atomic.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示CUDA原子操作，包括原子计数器和生产者-消费者同步模式
 * @version 0.1
 * @date 2026-03-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cuda/atomic>
#include <cuda_runtime_api.h>

__global__ void block_scoped_counter()
{
    // 仅在此块内可见的共享原子计数器
    __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

    // 初始化计数器(仅一个线程应该执行此操作)
    if (threadIdx.x == 0)
    {
        counter.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // 块内所有线程原子性地递增
    int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);

    // 使用old_value...
}

__global__ void producer_consumer()
{
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;

    if (threadIdx.x == 0)
    {
        // 生产者：写入数据然后发出准备信号
        data = 42;
        ready.store(true, cuda::memory_order_release); // Release确保数据写入可见
    }
    else
    {
        // 消费者：等待准备信号然后读取数据
        while (!ready.load(cuda::memory_order_acquire))
        { // Acquire确保数据读取看到写入
          // 自旋等待
        }
        int value = data;
        // 处理value...
    }
}