__global__ void reduce0(float* input, float* output, int N)
{
    // Calculate the index of the thread
    float* block_input = input + blockIdx.x * blockDim.x;

    /**
     // 1st
    if (threadIdx.x % 2 == 0)
            block_input[threadIdx.x] += block_input[threadIdx.x + 1];
    // 2nd
    if(threadIdx.x % 4 == 0)
            block_input[threadIdx.x] += block_input[threadIdx.x + 2];
    // 3rd
    if(threadIdx.x % 8 == 0)
            block_input[threadIdx.x] += block_input[threadIdx.x + 4];

    */
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (2 * i) == 0)
            block_input[threadIdx.x] += block_input[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = block_input[0];
}

// use shared memory
__global__ void reduce1(const float* input, float* output, int N)
{
    extern __shared__ float shared[];

    const float* block_input = input + blockIdx.x * blockDim.x;
    shared[threadIdx.x]      = block_input[threadIdx.x];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x % (2 * i) == 0)
            shared[threadIdx.x] += shared[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = shared[0];
}

// warp reduction
__global__ void reduce2(const float* input, float* output, int N)
{
    extern __shared__ float shared[];

    const float* block_input = input + blockIdx.x * blockDim.x;
    shared[threadIdx.x]      = block_input[threadIdx.x];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x < blockDim.x / (2 * i))
        {
            int index = threadIdx.x * 2 * i;
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = shared[0];
}

// no bank conflict
__global__ void reduce3(const float* input, float* output, int N)
{
    extern __shared__ float shared[];

    const float* block_input = input + blockIdx.x * blockDim.x;
    shared[threadIdx.x]      = block_input[threadIdx.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
            shared[threadIdx.x] += shared[threadIdx.x + i];

        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x] = shared[0];
}

__global__ void reduce5(const float* input, float* output, int N)
{
    extern volatile __shared__ float shared5[];

    const float* block_input = input + blockIdx.x * blockDim.x;
    shared5[threadIdx.x]     = block_input[threadIdx.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (threadIdx.x < i)
            shared5[threadIdx.x] += shared5[threadIdx.x + i];
        __syncthreads();
    }
    if (threadIdx.x < 32)
    {
        shared5[threadIdx.x] += shared5[threadIdx.x + 32];
        shared5[threadIdx.x] += shared5[threadIdx.x + 16];
        shared5[threadIdx.x] += shared5[threadIdx.x + 8];
        shared5[threadIdx.x] += shared5[threadIdx.x + 4];
        shared5[threadIdx.x] += shared5[threadIdx.x + 2];
        shared5[threadIdx.x] += shared5[threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = shared5[0];
}

__global__ void reduce6(const float* input, float* output, int N)
{
    extern volatile __shared__ float shared5[];

    const float* block_input = input + blockIdx.x * blockDim.x;
    shared5[threadIdx.x]     = block_input[threadIdx.x];
    __syncthreads();

#pragma unroll
    for (int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (threadIdx.x < i)
            shared5[threadIdx.x] += shared5[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        shared5[threadIdx.x] += shared5[threadIdx.x + 32];
        shared5[threadIdx.x] += shared5[threadIdx.x + 16];
        shared5[threadIdx.x] += shared5[threadIdx.x + 8];
        shared5[threadIdx.x] += shared5[threadIdx.x + 4];
        shared5[threadIdx.x] += shared5[threadIdx.x + 2];
        shared5[threadIdx.x] += shared5[threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = shared5[0];
}