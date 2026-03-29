/**
 * @file cooperative_groups.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief
 * @version 0.1
 * @date 2026-03-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

void test_create_groups()
{
    namespace cg = cooperative_groups;
    // Obtain the current thread's cooperative group
    cg::thread_block my_group = cg::this_thread_block();

    // Partition the cooperative group into tiles of size 8
    cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(my_group);

    // do work as my_subgroup

    // Synchronize threads in the block
    cg::sync(my_group);
}