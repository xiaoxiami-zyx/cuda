#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 这里模拟一个普通的非CUDA文件中定义的外部数据
char*             ext_data;
static const char global_string[] = "Hello World";

// 程序启动时分配并初始化外部数据
void __attribute__((constructor)) setup(void)
{
    ext_data = (char*)malloc(sizeof(global_string));
    strncpy(ext_data, global_string, sizeof(global_string));
}

// 程序退出时释放外部数据
void __attribute__((destructor)) tear_down(void)
{ free(ext_data); }