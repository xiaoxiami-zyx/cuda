/**
 * @file unified_memory.cu
 * @author yxzhou (1320162412@qq.com)
 * @brief 演示不同内存类型在CUDA Kernel中能否被直接访问，并对比统一内存的用法
 * @version 0.1
 * @date 2026-03-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <cassert>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define ASSERT(expr, ...)                                                                          \
    do                                                                                             \
    {                                                                                              \
        if (!(expr))                                                                               \
        {                                                                                          \
            fprintf(stderr, __VA_ARGS__);                                                          \
            assert(expr);                                                                          \
        }                                                                                          \
    } while (0)

__global__ void kernel(const char* type, const char* data)
{
    static const int n_char = 8;
    printf("%s - first %d characters: '", type, n_char);
    for (int i = 0; i < n_char; ++i)
        printf("%c", data[i]);
    printf("'\n");
}

// 具有完整 CUDA
// 统一内存支持的系统允许设备访问与该设备交互的主机进程所拥有的任何内存。

/**
 * @brief 使用主机堆内存分配的数据测试GPU访问能力
 *
 * 该示例先在CPU端通过malloc分配并拷贝字符串，再直接将指针传给Kernel。
 * 这种方式通常不能被GPU安全访问，用于说明普通主机内存与统一内存的区别。
 */
void test_malloc()
{
    const char test_string[] = "Hello World";
    char*      heap_data     = (char*)malloc(sizeof(test_string));
    strncpy(heap_data, test_string, sizeof(test_string));
    kernel<<<1, 1>>>("malloc", heap_data);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
    free(heap_data);
}

/**
 * @brief 使用统一内存分配的数据测试GPU访问能力
 *
 * 统一内存由cudaMallocManaged分配，CPU和GPU都可以访问，同一个指针可直接
 * 传入Kernel使用，是本文件想展示的重点用法。
 */
void test_managed()
{
    const char test_string[] = "Hello World";
    char*      data;
    cudaMallocManaged(&data, sizeof(test_string));
    strncpy(data, test_string, sizeof(test_string));
    kernel<<<1, 1>>>("managed", data);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
    cudaFree(data);
}

/**
 * @brief 使用栈上数组测试Kernel是否可以访问主机局部变量
 *
 * 该示例将栈变量直接传给Kernel，用于演示主机栈内存与GPU访问权限的差异。
 */
void test_stack()
{
    const char test_string[] = "Hello World";
    kernel<<<1, 1>>>("stack", test_string);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
}

/**
 * @brief 使用静态局部数组测试Kernel访问主机静态存储区数据
 *
 * 静态局部数组具有静态存储期，可用于说明主机静态内存与Kernel访问的关系。
 */
void test_static()
{
    static const char test_string[] = "Hello World";
    kernel<<<1, 1>>>("static", test_string);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
}

const char global_string[] = "Hello World";

/**
 * @brief 使用全局静态数组测试Kernel访问主机侧全局数据
 */
void test_global()
{
    kernel<<<1, 1>>>("global", global_string);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
}

// 声明来自其他文件的外部数据指针，便于演示跨文件共享数据
extern char* ext_data;

/**
 * @brief 使用外部全局指针测试Kernel访问其他文件准备的数据
 */
void test_extern()
{
    kernel<<<1, 1>>>("extern", ext_data);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'\n",
        cudaGetErrorString(cudaGetLastError()));
}

#define INPUT_FILE_NAME "input.txt"
void test_file_backed()
{
    int fd = open(INPUT_FILE_NAME, O_RDONLY);
    ASSERT(fd >= 0, "Invalid file handle");
    struct stat file_stat;
    int         status = fstat(fd, &file_stat);
    ASSERT(status >= 0, "Invalid file stats");
    char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
    kernel<<<1, 1>>>("file-backed", mapped);
    ASSERT(
        cudaDeviceSynchronize() == cudaSuccess,
        "CUDA failed with '%s'",
        cudaGetErrorString(cudaGetLastError()));
    ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
    // ASSERT(close(fd) == 0, "Cannot close file");
}

int main()
{
    // test_malloc();
    test_managed();
    // test_stack();
    // test_static();
    // test_global();
    // test_extern();

    return 0;
}