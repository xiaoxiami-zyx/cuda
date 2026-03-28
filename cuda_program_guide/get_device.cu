/**
 * @file get_device.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief
 * @version 0.1
 * @date 2026-03-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void MyKernel(float* data)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 0.5f; // Just an example operation
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf(
            "Device %d (%s) has compute capability %d.%d.\n",
            device,
            deviceProp.name,
            deviceProp.major,
            deviceProp.minor);
    }

    // 选择设备0进行计算
    {
        size_t size = 1024 * sizeof(float);
        cudaSetDevice(0); // Set device 0 as current
        float* p0;
        cudaMalloc(&p0, size);       // Allocate memory on device 0
        MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

        cudaSetDevice(1); // Set device 1 as current
        float* p1;
        cudaMalloc(&p1, size);       // Allocate memory on device 1
        MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
    }

    // 设备间通信
    {
        cudaSetDevice(0); // Set device 0 as current
        float* p0;
        size_t size = 1024 * sizeof(float);
        cudaMalloc(&p0, size); // Allocate memory on device 0

        cudaSetDevice(1); // Set device 1 as current
        float* p1;
        cudaMalloc(&p1, size); // Allocate memory on device 1

        cudaSetDevice(0);            // Set device 0 as current
        MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

        cudaSetDevice(1);                   // Set device 1 as current
        cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
        MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
    }

    // 设备间点对点内存访问
    {
        cudaSetDevice(0); // Set device 0 as current
        float* p0;
        size_t size = 1024 * sizeof(float);
        cudaMalloc(&p0, size);       // Allocate memory on device 0
        MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

        cudaSetDevice(1);                 // Set device 1 as current
        cudaDeviceEnablePeerAccess(0, 0); // Enable peer-to-peer access
                                          // with device 0

        // Launch kernel on device 1
        // This kernel launch can access memory on device 0 at address p0
        MyKernel<<<1000, 128>>>(p0);
    }
}