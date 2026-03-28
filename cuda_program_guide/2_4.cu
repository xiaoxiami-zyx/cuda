#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <iostream>

// 查询设备属性以确定统一内存支持级别
void queryDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; i++)
    {
        cudaSetDevice(i);
        cudaInitDevice(0, 0, 0);
        int deviceId = i;

        int concurrentManagedAccess = -1;
        cudaDeviceGetAttribute(
            &concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);
        int pageableMemoryAccess = -1;
        cudaDeviceGetAttribute(&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
        int pageableMemoryAccessUsesHostPageTables = -1;
        cudaDeviceGetAttribute(
            &pageableMemoryAccessUsesHostPageTables,
            cudaDevAttrPageableMemoryAccessUsesHostPageTables,
            deviceId);

        printf("Device %d has ", deviceId);
        if (concurrentManagedAccess)
        {
            if (pageableMemoryAccess)
            {
                printf("full unified memory support");
                if (pageableMemoryAccessUsesHostPageTables)
                {
                    printf(" with hardware coherency\n");
                }
                else
                {
                    printf(" with software coherency\n");
                }
            }
            else
            {
                printf("full unified memory support for CUDA-made managed allocations\n");
            }
        }
        else
        {
            printf("limited unified memory support: Windows, WSL, or Tegra\n");
        }
    }
}

int main()
{
    int* a;
    int* d_a;

    int N = 100;

    // Allocate memory on the host
    a = (int*)malloc(N * sizeof(int));
    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(int));

    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, d_a);

    std::cout << "Device pointer attributes:" << std::endl;
    std::cout << "Type: " << attr.type << std::endl;
    std::cout << "Device: " << attr.device << std::endl;
    std::cout << "Host pointer: " << attr.hostPointer << std::endl;
    std::cout << "Device pointer: " << attr.devicePointer << std::endl;

    queryDevices();

    free(a);
    cudaFree(d_a);

    return 0;
}