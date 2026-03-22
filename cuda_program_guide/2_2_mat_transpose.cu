#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <stdio.h>

const int numCols = 32;
const int numRows = 32;

__global__ void mat_transpose(const float* A, float* A_T)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numRows)
    {
        A_T[col * numRows + row] = A[row * numRows + col];
    }
}

__global__ void mat_transpose_shared(const float* A, float* A_T)
{
    __shared__ float tile[32][33];

    int x = blockIdx.x * blockDim.x + threadIdx.x; // input col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // input row

    if (y < numRows && x < numCols)
    {
        tile[threadIdx.y][threadIdx.x] = A[y * numCols + x];
    }

    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x; // output col
    y = blockIdx.x * blockDim.x + threadIdx.y; // output row

    if (y < numCols && x < numRows)
    {
        A_T[y * numRows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    float* A   = (float*)malloc(numRows * numCols * sizeof(float));
    float* A_T = (float*)malloc(numRows * numCols * sizeof(float));

    float* A_device   = nullptr;
    float* A_T_device = nullptr;

    cudaMalloc(&A_device, numRows * numCols * sizeof(float));
    cudaMalloc(&A_T_device, numRows * numCols * sizeof(float));

    for (int i = 0; i < numRows * numCols; ++i)
    {
        A[i] = static_cast<float>(i);
    }

    cudaMemcpy(A_device, A, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    mat_transpose_shared<<<1, blockSize>>>(A_device, A_T_device);

    cudaMemcpy(A_T, A_T_device, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            if (A_T[i * numCols + j] != A[j * numRows + i])
            {
                printf(
                    "Error at position (%d, %d): expected %f, got %f\n",
                    i,
                    j,
                    A[j * numRows + i],
                    A_T[i * numCols + j]);
            }
        }
    }

    free(A);
    free(A_T);
    cudaFree(A_device);
    cudaFree(A_T_device);

    return 0;
}