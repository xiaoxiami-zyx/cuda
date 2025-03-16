#include <cuda_runtime.h>
#if 1

// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float* input, float* output) {}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size

    int N = 1024 * 1024;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}

#else

// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float* input, float* output) {}

int main()
{
    int    N = 1024 * 1024;
    float *input, *output;
    dim3   threadsPerBlock(16, 16);
    dim3   numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim  = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs                  = attribute;
        config.numAttrs               = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}

#endif