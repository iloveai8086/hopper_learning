// This code demonstrate on sm_90 GPU,
// how to create a cluster of thread blocks
// and how blocks in a cluster can interact
// using distributed shared memory.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "profile_utilities.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK CHECK_CUDA
#endif

__global__ void __cluster_dims__(2, 1, 1) cluster_kernel() {
	// printf("blockIdx.x: %d, threadIdx.x: %d\n", blockIdx.x, threadIdx.x);

	__shared__ int smem[32];
	namespace cg = cooperative_groups;

	// tid is the thread index within the cluster, not block.
	int tid = cg::this_grid().thread_rank();

	cg::cluster_group cluster = cg::this_cluster();
	unsigned int clusterBlockRank = cluster.block_rank();
	int cluster_size = cluster.dim_blocks().x;

	// cluster size = nubmer of blocks in the cluster
	if (tid == 0) {
		printf("cluster_size: %d\n", cluster_size);
	}

	// initialize shared memory, block 1 has one value higher than block 0
	smem[threadIdx.x] = blockIdx.x + threadIdx.x;

	cluster.sync();

	// get the shared memory of the other block
	int *other_block_smem = cluster.map_shared_rank(smem, 1 - clusterBlockRank);

	// get the value from the other block
	int value = other_block_smem[threadIdx.x];

	cluster.sync();

	// print the value
	printf("blockIdx.x: %d, threadIdx.x: %d, value: %d\n", blockIdx.x,
		   threadIdx.x, value);
}

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins,
								   const int bins_per_block,
								   const int *__restrict__ input,
								   size_t array_size) {
	extern __shared__ int smem[];
	namespace cg = cooperative_groups;
	int tid = cg::this_grid().thread_rank();

	// Cluster initialization, size and calculating local bin offsets.
	cg::cluster_group cluster = cg::this_cluster();
	unsigned int clusterBlockRank = cluster.block_rank();
	int cluster_size = cluster.dim_blocks().x;

	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		smem[i] = 0; // Initialize shared memory histogram to zeros
	}

	// cluster synchronization ensures that shared memory is initialized to zero
	// in all thread blocks in the cluster. It also ensures that all thread
	// blocks have started executing and they exist concurrently.
	cluster.sync();

	for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
		int ldata = input[i];

		// Find the right histogram bin.
		int binid = ldata;
		if (ldata < 0)
			binid = 0;
		else if (ldata >= nbins)
			binid = nbins - 1;

		// Find destination block rank and offset for computing
		// distributed shared memory histogram
		int dst_block_rank = (int)(binid / bins_per_block);
		int dst_offset = binid % bins_per_block;

		// Pointer to target block shared memory
		int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

		// Perform atomic update of the histogram bin
		atomicAdd(dst_smem + dst_offset, 1);
	}

	// cluster synchronization is required to ensure all distributed shared
	// memory operations are completed and no thread block exits while
	// other thread blocks are still accessing distributed shared memory
	cluster.sync();

	// Perform global memory histogram, using the local distributed memory
	// histogram
	int *lbins = bins + cluster.block_rank() * bins_per_block;
	for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
		atomicAdd(&lbins[i], smem[i]);
	}
}

__global__ void dummy_kernel() { /* no-op */ }

#define CHECK(call)                                                            \
	do {                                                                       \
		cudaError_t _e = (call);                                               \
		if (_e != cudaSuccess) {                                               \
			fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
					cudaGetErrorString(_e), __FILE__, __LINE__);               \
			return 1;                                                          \
		}                                                                      \
	} while (0)

int main() {

	// define and allocate variables required by the extensible launch example
	const int nbins = 64;
	const int array_size = 1 << 12;
	const int threads_per_block = 128;
	int *bins = nullptr;
	int *input = nullptr;
	CHECK_CUDA(cudaMalloc(&bins, nbins * sizeof(int)));
	CHECK_CUDA(cudaMemset(bins, 0, nbins * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&input, array_size * sizeof(int)));
	CHECK_CUDA(cudaMemset(input, 0, array_size * sizeof(int)));

	// two blocks in a cluster
	cluster_kernel<<<2, 32>>>();

	cuda_check_error();

	// Launch via extensible launch
	{
		cudaLaunchConfig_t config = {0};
		config.gridDim = array_size / threads_per_block;
		config.blockDim = threads_per_block;

		// cluster_size depends on the histogram size.
		// ( cluster_size == 1 ) implies no distributed shared memory, just
		// thread block local shared memory
		int cluster_size = 2; // size 2 is an example here
		int nbins_per_block = nbins / cluster_size;

		// dynamic shared memory size is per block
		// Distributed shared memory size =  cluster_size * nbins_per_block *
		// sizeof(int)
		config.dynamicSmemBytes = nbins_per_block * sizeof(int);

		CUDA_CHECK(
			::cudaFuncSetAttribute((void *)clusterHist_kernel,
								   cudaFuncAttributeMaxDynamicSharedMemorySize,
								   config.dynamicSmemBytes));

		cudaLaunchAttribute attribute[1];
		attribute[0].id = cudaLaunchAttributeClusterDimension;
		attribute[0].val.clusterDim.x = cluster_size;
		attribute[0].val.clusterDim.y = 1;
		attribute[0].val.clusterDim.z = 1;

		config.numAttrs = 1;
		config.attrs = attribute;

		cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins,
						   nbins_per_block, input, array_size);
	}

	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(bins));
	CHECK_CUDA(cudaFree(input));

	std::cout << "Device " << std::endl;

	{
		int dev = 0;
		CHECK(cudaGetDevice(&dev));
		cudaDeviceProp prop{};
		CHECK(cudaGetDeviceProperties(&prop, dev));
		printf("Device %d: %s, CC %d.%d\n", dev, prop.name, prop.major,
			   prop.minor);

		// 查询设备支持的 cluster 大小
		int maxCluster = 0;
		cudaLaunchConfig_t config = {0};
		config.gridDim = 1;
		config.blockDim = 32;
		
		cudaError_t err = cudaOccupancyMaxPotentialClusterSize(&maxCluster, dummy_kernel, &config);
		if (err == cudaSuccess) {
			printf("Max cluster size: %d\n", maxCluster);
		} else {
			printf("Cluster size query not supported: %s\n", cudaGetErrorString(err));
		}
	}
	return 0;
}
