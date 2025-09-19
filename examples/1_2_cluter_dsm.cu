#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "profile_utilities.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK CHECK_CUDA
#endif

// 方法1：传统 <<<>>> 语法调用（编译时指定 cluster 大小）
__global__ void __cluster_dims__(2, 1, 1)
	clusterHist_kernel_traditional(int *bins, const int nbins,
								   const int bins_per_block,
								   const int *__restrict__ input,
								   size_t array_size) {
	extern __shared__ int smem[];
	namespace cg = cooperative_groups;
	int tid = cg::this_grid().thread_rank();

	// Cluster initialization, size and calculating local bin offsets.
	// auto cluster = cg::this_thread_block().cluster();
	cg::cluster_group cluster = cg::this_cluster();
	unsigned int clusterBlockRank = cluster.block_rank();
	int cluster_size = cluster.dim_blocks().x;

	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		smem[i] = 0; // Initialize shared memory histogram to zeros
	}

	cluster.sync();

	for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
		int ldata = input[i];
		int binid = ldata;
		if (ldata < 0)
			binid = 0;
		else if (ldata >= nbins)
			binid = nbins - 1;

		int dst_block_rank = (int)(binid / bins_per_block);
		int dst_offset = binid % bins_per_block;
		int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
		atomicAdd(dst_smem + dst_offset, 1);
	}

	cluster.sync();

	int *lbins = bins + cluster.block_rank() * bins_per_block;
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		atomicAdd(&lbins[i], smem[i]);
	}
}

// 方法2：Extensible Launch API 调用的 kernel
__global__ void clusterHist_kernel_extensible(int *bins, const int nbins,
											  const int bins_per_block,
											  const int *__restrict__ input,
											  size_t array_size) {
	extern __shared__ int smem[];
	namespace cg = cooperative_groups;
	int tid = cg::this_grid().thread_rank();

	// Cluster initialization, size and calculating local bin offsets.
	// auto cluster = cg::this_thread_block().cluster();
	cg::cluster_group cluster = cg::this_cluster();  // 在这里等价于 整个 grid 内所有线程的全局线性线程 ID。int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int clusterBlockRank = cluster.block_rank();  // 0 or 1，这个变量就是当前cluster内的block的rank，rank是就是block的index
	int cluster_size = cluster.dim_blocks().x;  // 2

	// printf("clusterBlockRank: %d\n", clusterBlockRank);
	// printf("cluster_size: %d\n", cluster_size);

	// bins_per_block为32，blockDim.x为128，所以这个循环在干什么？我觉得应该就是防止bins_per_block>blockDim.x的情况，需要循环初始化
	// smem的大小就是bins_per_block，32个block，每个block 128个线程
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		smem[i] = 0; // Initialize shared memory histogram to zeros
	}

	// cluster 同步确保簇内所有线程块的共享内存都被初始化为零。
	// 它还保证簇内所有线程块都已经开始执行，并且它们是并发存在的。
	// cluster synchronization ensures that shared memory is initialized to zero
	// in all thread blocks in the cluster. It also ensures that all thread
	// blocks have started executing and they exist concurrently.
	cluster.sync();

	for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
		int ldata = input[i];
		// Find the right histogram bin. 有效 bin 下标范围是 [0, nbins-1] 如果数据落在范围外，就把它归到边界 bin（第一个或最后一个）。
		int binid = ldata;
		if (ldata < 0)
			binid = 0;
		else if (ldata >= nbins)
			binid = nbins - 1;

		// 查找目标线程块的 rank 和在该块内的偏移量，用于计算
		// 分布式共享内存直方图。
		// Find destination block rank and offset for computing
		// distributed shared memory histogram
		int dst_block_rank = (int)(binid / bins_per_block);  // 当前的bin在哪个block
		int dst_offset = binid % bins_per_block;  // 在某一个block内的具体偏移
		// Pointer to target block shared memory 指向目标线程块共享内存的指针
		int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
		// Perform atomic update of the histogram bin 执行对直方图 bin 的原子更新
		atomicAdd(dst_smem + dst_offset, 1);
	}

	// 需要进行 cluster 同步，以确保所有分布式共享内存操作完成，
	// 并且在其他线程块仍在访问分布式共享内存时，没有线程块提前退出
	// cluster synchronization is required to ensure all distributed shared
	// memory operations are completed and no thread block exits while
	// other thread blocks are still accessing distributed shared memory
	cluster.sync();

	// cluster里面的block都做完后，统一的把整个cluster内的都加起来，然后更新到全局内存中
	// 使用本地分布式共享内存直方图，来执行全局内存直方图的更新
	// Perform global memory histogram, using the local distributed memory
	// histogram
	int *lbins = bins + cluster.block_rank() * bins_per_block;  // 这边的bin，由于block的rank不同，取值范围是0-63
	// bins + (0-1) * 32 = 0-63
	// 线程id是0-127
	// 第一个block很好理解，关键是第二个block，此时i仍然是0-31，lbins的地址已经被加上32了，所以此时的偏移是对的
	// 而第二个block内的smem也是正确的，所以可以直接加
	// 核心点就是按照不同的block的smem来做
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		atomicAdd(&lbins[i], smem[i]);
	}
}

void cpu_histogram(int *bins, const int nbins, const int *input, const int array_size) {
	for (int i = 0; i < array_size; i++) {
		bins[input[i]]++;
	}
}

int main() {
	// 定义测试数据
	const int nbins = 64;
	const int array_size = 1 << 12;  // 4096 * 1的数组？
	const int threads_per_block = 128;
	const int cluster_size = 2;  // 每个cluster里面2个block
	const int nbins_per_block = nbins / cluster_size;  // 64 / 2 = 32 计算出每一个block里面的bin的数量

	int *bins = nullptr;
	int *input = nullptr;

	// 分配内存
	CHECK_CUDA(cudaMalloc(&bins, nbins * sizeof(int)));
	CHECK_CUDA(cudaMemset(bins, 0, nbins * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&input, array_size * sizeof(int)));

	// 初始化输入数据
	int *h_input = new int[array_size];
	for (int i = 0; i < array_size; i++) {
		h_input[i] = i % nbins; // 生成 0 到 nbins-1 的随机数据，4096个数据，每个数据在0-63之间
	}
	CHECK_CUDA(cudaMemcpy(input, h_input, array_size * sizeof(int),
						  cudaMemcpyHostToDevice));

	printf("=== 方法1：传统 <<<>>> 语法调用 ===\n");
	// 方法1：传统调用方式
	clusterHist_kernel_traditional<<<array_size / threads_per_block,  // 4096 / 128 = 32个block
									 threads_per_block,
									 nbins_per_block * sizeof(int)>>>(
		bins, nbins, nbins_per_block, input, array_size);
	CHECK_CUDA(cudaDeviceSynchronize());

	// 验证结果
	int *h_bins = new int[nbins];
	CHECK_CUDA(
		cudaMemcpy(h_bins, bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));

	int total_count = 0;
	for (int i = 0; i < nbins; i++) {
		total_count += h_bins[i];
	}
	printf("传统方法 - 总计数: %d (期望: %d)\n", total_count, array_size);
	// 和cpu对比
	int *h_bins_cpu = new int[nbins];
	memset(h_bins_cpu, 0, nbins * sizeof(int));  // 初始化CPU结果数组
	cpu_histogram(h_bins_cpu, nbins, h_input, array_size);
	bool all_match = true;
	for (int i = 0; i < nbins; i++) {
		if (h_bins[i] != h_bins_cpu[i]) {
			printf("Error at bin %d: GPU=%d, CPU=%d\n", i, h_bins[i], h_bins_cpu[i]);
			all_match = false;
		}
	}
	if (all_match) {
		printf("CPU和GPU的计算结果一致\n");
	} else {
		printf("CPU和GPU的计算结果不一致\n");
	}

	// 重置 bins
	CHECK_CUDA(cudaMemset(bins, 0, nbins * sizeof(int)));

	printf("\n=== 方法2：Extensible Launch API 调用 ===\n");
	// 方法2：Extensible Launch API
	{
		cudaLaunchConfig_t config = {0};
		config.gridDim = array_size / threads_per_block;
		config.blockDim = threads_per_block;
		config.dynamicSmemBytes = nbins_per_block * sizeof(int);

		CUDA_CHECK(
			::cudaFuncSetAttribute((void *)clusterHist_kernel_extensible,
								   cudaFuncAttributeMaxDynamicSharedMemorySize,
								   config.dynamicSmemBytes));

		cudaLaunchAttribute attribute[1];
		attribute[0].id = cudaLaunchAttributeClusterDimension;
		attribute[0].val.clusterDim.x = cluster_size;
		attribute[0].val.clusterDim.y = 1;
		attribute[0].val.clusterDim.z = 1;

		config.numAttrs = 1;
		config.attrs = attribute;

		cudaLaunchKernelEx(&config, clusterHist_kernel_extensible, bins, nbins,
						   nbins_per_block, input, array_size);
	}

	CHECK_CUDA(cudaDeviceSynchronize());

	// 验证结果
	CHECK_CUDA(
		cudaMemcpy(h_bins, bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));

	total_count = 0;
	for (int i = 0; i < nbins; i++) {
		total_count += h_bins[i];
	}
	printf("Extensible Launch - 总计数: %d (期望: %d)\n", total_count,
		   array_size);

	// 清理内存
	CHECK_CUDA(cudaFree(bins));
	CHECK_CUDA(cudaFree(input));
	delete[] h_input;
	delete[] h_bins;

	printf("\n两种调用方法都执行完成！\n");
	return 0;
}