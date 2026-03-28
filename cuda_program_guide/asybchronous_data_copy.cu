/**
 * @file asybchronous_data_copy.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示有无异步拷贝两种方式下，分批将全局内存数据搬运到共享内存并进行计算的流程
 * @version 0.1
 * @date 2026-03-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in)
{
    // 使用当前批次在共享内存中的全部数据进行计算。
    // 将当前线程的结果写回全局内存。
}

/**
 * @brief 同步拷贝版本：将每个批次的数据从全局内存搬运到共享内存后再计算
 *
 * 每轮批次中，各线程先把对应元素写入共享内存，再通过block.sync()保证
 * 全部数据可见，随后调用compute()完成本批次计算并写回global_out。
 *
 * @param global_out 输出数组首地址，用于写回每个批次的计算结果
 * @param global_in 输入数组首地址，按批次读取原始数据
 * @param size 输入总元素个数，需满足size == batch_sz * grid.size()
 * @param batch_sz 批次数量
 */
__global__ void
without_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid  = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // 说明：输入大小应满足batch_sz * grid_size

    extern __shared__ int shared[]; // 共享内存大小为block.size() * sizeof(int)字节

    size_t local_idx = block.thread_rank();

    for (size_t batch = 0; batch < batch_sz; ++batch)
    {
        // 计算该线程块在当前批次中的全局内存索引。
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
        size_t global_idx      = block_batch_idx + threadIdx.x;
        shared[local_idx]      = global_in[global_idx];

        // 等待所有线程完成数据拷贝。
        block.sync();

        // 进行计算并将结果写回全局内存。
        compute(global_out + block_batch_idx, shared);

        // 等待所有线程完成基于共享内存的计算。
        block.sync();
    }
}

/**
 * @brief 异步拷贝版本：使用cooperative_groups::memcpy_async协同搬运并计算
 *
 * 每轮批次中，线程组先发起全局内存到共享内存的异步拷贝，随后可在等待期间
 * 执行其他计算；在cooperative_groups::wait()后保证数据可用，再调用compute()。
 *
 * @param global_out 输出数组首地址，用于写回每个批次的计算结果
 * @param global_in 输入数组首地址，按批次读取原始数据
 * @param size 输入总元素个数，需满足size == batch_sz * grid.size()
 * @param batch_sz 批次数量
 */
__global__ void with_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz)
{
    auto grid  = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // 说明：输入大小应满足batch_sz * grid_size

    extern __shared__ int shared[]; // 共享内存大小为block.size() * sizeof(int)字节

    size_t local_idx = block.thread_rank();

    for (size_t batch = 0; batch < batch_sz; ++batch)
    {
        // 计算该线程块在当前批次中的全局内存索引。
        size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

        // 整个线程组协同将当前批次数据拷贝到共享内存。
        cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, block.size());

        // 等待期间可以对其他数据进行计算。

        // 等待所有异步拷贝完成。
        cooperative_groups::wait(block);

        // 进行计算并将结果写回全局内存。
        compute(global_out + block_batch_idx, shared);

        // 等待所有线程完成基于共享内存的计算。
        block.sync();
    }
}