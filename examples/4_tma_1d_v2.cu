/*
This code uses TMA's 1d tensor load to load
a portion of an array to shared memory and then
change the value in the shared memory and uses TMA's store
to store the portion back to global memory. We print the result
to show the changes are done.
*/

// supress warning about barrier in shared memory on line 32
#pragma nv_diag_suppress static_var_with_dynamic_init

#include <cuda/barrier>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cuda/barrier>

#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"
#include "tma_tensor_map.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int *data, size_t offset) {
	// Shared memory 数组。数组整体 size 要对齐 16字节
	__shared__ alignas(16) int smem_data[buf_len];

// 1. a) 用0号线程初始化 barrier，与上面的代码示例类似。
//    b) 插入一个fence。表示后续执行异步拷贝操作，需要在这个fence之后才执行。
#pragma nv_diag_suppress static_var_with_dynamic_init
	__shared__ barrier bar;
	if (threadIdx.x == 0) {
		init(&bar, blockDim.x);										// a)
		cuda::device::experimental::fence_proxy_async_shared_cta(); // b)
	}
	__syncthreads();

	// 2. 发起 TMA 异步拷贝。注意：TMA 操作是用单线程发起。
	if (threadIdx.x == 0) {
		// 3a. 发起异步拷贝
		cuda::memcpy_async(smem_data, data + offset,
						   cuda::aligned_size_t<16>(sizeof(smem_data)), bar);
	}
	// 3b. 所有线程到达该标记点，barrier内部的计数器会加 1。
	barrier::arrival_token token = bar.arrive();

	// 3c.等待barrier内部的计数器等于期望数值，即所有线程到达3b点时，当前线程的wait会返回，结束等待，可以和上面何为arrive_and_wait
	bar.wait(std::move(token));

	// 4. 在 Shared Memory 上写数据。
	for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
		smem_data[i] += 1;
	}

	// 5. 插入fence，使得修改对TMA proxy可见， Wait for shared memory writes to
	// be visible to TMA engine.
	cuda::device::experimental::fence_proxy_async_shared_cta(); // b)

	__syncthreads();
	// After syncthreads, writes by all threads are visible to TMA engine.

	// 6. 发起从 Shared Memory 到 Global Memory 的异步拷贝操作。
	if (threadIdx.x == 0) {
		cuda::device::experimental::cp_async_bulk_shared_to_global(
			data + offset, smem_data, sizeof(smem_data));
		// 7. 一种同步方式，创建一个 bulk async-group，异步拷贝在这个 group
		// 中运行，当异步拷贝结束后， group 内部标记为已完成。
		cuda::device::experimental::cp_async_bulk_commit_group();
		// 等待 group 完成。模版参数 0 表示要等待小于等于 0 个 bulk async-group
		// 完成才结束等待。
		cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
	}
}

int main() {
	// 设置随机种子
	srand(42);
	
	// 数据大小配置
	constexpr size_t data_size = 2048;  // 总数据大小
	constexpr size_t block_size = 128;  // 线程块大小
	constexpr size_t num_blocks = (data_size + buf_len - 1) / buf_len;  // 需要的块数
	
	printf("TMA 1D 示例程序\n");
	printf("数据大小: %zu\n", data_size);
	printf("缓冲区大小: %zu\n", buf_len);
	printf("线程块大小: %zu\n", block_size);
	printf("块数量: %zu\n", num_blocks);
	
	// 1. 分配主机内存
	int *h_data = new int[data_size];
	
	// 2. 初始化主机数据
	printf("\n初始化主机数据...\n");
	for (size_t i = 0; i < data_size; i++) {
		h_data[i] = i;  // 初始化为索引值
	}
	
	// 打印前16个元素
	printf("原始数据前16个元素: ");
	for (int i = 0; i < 16; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n");
	
	// 3. 分配设备内存
	int *d_data;
	CUDA_CHECK(cudaMalloc(&d_data, data_size * sizeof(int)));
	
	// 4. 复制数据到设备
	CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size * sizeof(int), cudaMemcpyHostToDevice));
	
	// 5. 创建计时器
	cuda_timer timer;
	timer.start_timer();
	
	// 6. 启动kernel
	printf("\n启动kernel...\n");
	for (size_t block_id = 0; block_id < num_blocks; block_id++) {
		size_t offset = block_id * buf_len;
		// 确保不会越界
		size_t actual_size = (offset + buf_len > data_size) ? (data_size - offset) : buf_len;
		
		if (actual_size > 0) {
			add_one_kernel<<<1, block_size>>>(d_data, offset);
		}
	}
	
	// 7. 同步设备
	cuda_check_error();
	timer.stop_timer();
	
	printf("Kernel执行完成，耗时: %.2f ms\n", timer.get_time());
	
	// 8. 复制结果回主机
	CUDA_CHECK(cudaMemcpy(h_data, d_data, data_size * sizeof(int), cudaMemcpyDeviceToHost));
	
	// 9. 验证结果
	printf("\n验证结果...\n");
	printf("处理后数据前16个元素: ");
	for (int i = 0; i < 16; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n");
	
	// 检查是否正确（每个元素应该增加1）
	bool correct = true;
	for (size_t i = 0; i < data_size; i++) {
		if (h_data[i] != (int)(i + 1)) {
			printf("错误: 位置 %zu 的值不正确，期望 %zu，实际 %d\n", 
				   i, i + 1, h_data[i]);
			correct = false;
			break;
		}
	}
	
	if (correct) {
		printf("✓ 所有元素都正确增加了1！\n");
	} else {
		printf("✗ 发现错误！\n");
	}
	
	// 10. 清理资源
	delete[] h_data;
	CUDA_CHECK(cudaFree(d_data));
	
	printf("\n程序执行完成！\n");
	return 0;
}
