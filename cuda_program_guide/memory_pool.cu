/**
 * @file memory_pool.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示CUDA显式内存池创建，包括设备内存池与CPU NUMA内存池
 * @version 0.1
 * @date 2026-03-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "error.cu"
#include <cuda_runtime_api.h>

__global__ void fill_kernel(int* data, int n, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = value;
}

/**
 * @brief 创建一个驻留在GPU设备上的显式内存池
 *
 * 该示例演示如何按设备位置配置cudaMemPoolProps，并创建/销毁内存池。
 */
void test_pool()
{
    cudaMemPool_t memPool = nullptr;

    // 创建一个与设备0隐式内存池属性相近的显式内存池。
    int              device    = 0;
    cudaMemPoolProps poolProps = {};
    poolProps.allocType        = cudaMemAllocationTypePinned;
    poolProps.location.id      = device;
    poolProps.location.type    = cudaMemLocationTypeDevice;

    CHECK_CUDA(cudaMemPoolCreate(&memPool, &poolProps));
    CHECK_CUDA(cudaMemPoolDestroy(memPool));
}

/**
 * @brief 创建一个驻留在CPU NUMA节点上且支持IPC句柄导出的内存池
 *
 * 该示例使用POSIX文件描述符句柄类型，便于后续跨进程共享场景。
 */
void test_cpu_numa_pool()
{
    cudaMemPool_t ipcMemPool = nullptr;

    // 创建一个驻留在CPU NUMA节点上的内存池，并通过文件描述符支持IPC共享。
    int              cpu_numa_id = 0;
    cudaMemPoolProps poolProps   = {};
    poolProps.allocType          = cudaMemAllocationTypePinned;
    poolProps.location.id        = cpu_numa_id;
    poolProps.location.type      = cudaMemLocationTypeHostNuma;
    poolProps.handleTypes        = cudaMemHandleTypePosixFileDescriptor;

    CHECK_CUDA(cudaMemPoolCreate(&ipcMemPool, &poolProps));
    CHECK_CUDA(cudaMemPoolDestroy(ipcMemPool));
}

/**
 * @brief 使用显式内存池进行异步分配、内核访问与异步释放
 *
 * 流程：创建设备内存池 -> 设为当前设备默认池 -> cudaMallocAsync分配 ->
 * kernel使用 -> cudaFreeAsync释放 -> 同步并销毁资源。
 */
void test_pool_alloc_usage()
{
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaMemPool_t customPool = nullptr;
    cudaMemPool_t oldPool    = nullptr;
    cudaStream_t  stream     = nullptr;

    cudaMemPoolProps props = {};
    props.allocType        = cudaMemAllocationTypePinned;
    props.location.type    = cudaMemLocationTypeDevice;
    props.location.id      = device;

    CHECK_CUDA(cudaMemPoolCreate(&customPool, &props));
    CHECK_CUDA(cudaDeviceGetDefaultMemPool(&oldPool, device));
    CHECK_CUDA(cudaDeviceSetMemPool(device, customPool));
    CHECK_CUDA(cudaStreamCreate(&stream));

    int*   d_data = nullptr;
    int    n      = 1 << 10;
    size_t bytes  = static_cast<size_t>(n) * sizeof(int);

    CHECK_CUDA(cudaMallocAsync(reinterpret_cast<void**>(&d_data), bytes, stream));

    int block = 256;
    int grid  = (n + block - 1) / block;
    fill_kernel<<<grid, block, 0, stream>>>(d_data, n, 7);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFreeAsync(d_data, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaDeviceSetMemPool(device, oldPool));
    CHECK_CUDA(cudaMemPoolDestroy(customPool));
}
