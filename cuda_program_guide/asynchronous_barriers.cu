/**
 * @file asynchronous _barriers.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示CUDA异步屏障用于线程同步的分离到达-等待机制
 * @version 0.1
 * @date 2026-03-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cooperative_groups.h>
#include <cuda/barrier>
__device__ void compute(float* data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float* data)
{
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier_t bar;
    auto                 block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0)
    {
        // 使用预期的到达线程数初始化屏障
        init(&bar, block.size());
    }
    block.sync();

    for (int i = 0; i < iteration_count; ++i)
    {
        /* 到达屏障前的代码 */

        // 该线程到达屏障。到达不会阻塞线程。
        barrier_t::arrival_token token = bar.arrive();

        compute(data, i);

        // 等待所有参与屏障的线程完成bar.arrive()。
        bar.wait(std::move(token));

        /* 等待后的代码 */
    }
}