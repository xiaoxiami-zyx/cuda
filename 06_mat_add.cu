#include <cuda_runtime.h>
#include <stdio.h>

void initData(float* p, int n)
{
    for (int i = 0; i < n; i++)
    {
        p[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void addFromGPU(float* a, float* b, float* c, size_t n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n)
        c[id] = a[id] + b[id];
}

int main()
{
    int iDevice = 0;
    cudaGetDeviceCount(&iDevice);
    printf("Number of CUDA devices: %d\n", iDevice);
    cudaSetDevice(0);

    const int N    = 1 << 25;
    size_t    size = N * sizeof(float);

    // 分配主机内存
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 分配cuda内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    srand(100);
    initData(h_A, N);
    initData(h_B, N);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 启动内核
    dim3 block(32);
    dim3 grid(N / 32);
    addFromGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 复制结果到主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 计算结果
    float sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += h_A[i] + h_B[i];
    }
    printf("CPU result: %f\n", sum);
    sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += h_C[i];
    }
    printf("GPU result: %f\n", sum);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Total global memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("compute capability: %d.%d\n", prop.major, prop.minor);
}