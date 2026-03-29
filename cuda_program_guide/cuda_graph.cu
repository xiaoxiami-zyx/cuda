/**
 * @file cuda_graph.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示如何手动创建CUDA Graph并配置内核节点之间的依赖关系
 * @version 0.1
 * @date 2026-03-29
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cuda_runtime_api.h>
#include <vector>

__global__ void kernelName()
{
    // 这是一个占位内核，实际使用时请替换为你的内核实现
}

int main()
{
    cudaGraph_t graph;
    // 创建一个空的CUDA Graph
    cudaGraphCreate(&graph, 0);

    // 创建图节点并配置内核参数(此处4个节点均使用同一个内核配置)
    cudaGraphNode_t     nodes[4];
    cudaGraphNodeParams kParams = {cudaGraphNodeTypeKernel};
    kParams.kernel.func         = (void*)kernelName;
    kParams.kernel.gridDim.x = kParams.kernel.gridDim.y = kParams.kernel.gridDim.z = 1;
    kParams.kernel.blockDim.x = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;

    // 节点0无前驱，作为起始节点
    cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);

    // 节点1依赖节点0(节点0完成后才能执行节点1)
    cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);

    // 节点2同样依赖节点0，可与节点1并行(前提是资源允许)
    cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);

    // 节点3同时依赖节点1和节点2，形成汇聚关系
    // 这里依赖数组从&nodes[1]开始，长度为2，即{nodes[1], nodes[2]}
    cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 2, &kParams);
}