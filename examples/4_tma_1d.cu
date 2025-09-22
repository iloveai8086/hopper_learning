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
#include <iostream>

#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"
#include "tma_tensor_map.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

const int array_size = 128 * 4;
const int tile_size = 16;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map,
					   int coordinate) {
	// Shared memory buffers for tile. The destination shared memory buffer of
	// a bulk operations should be 16 byte aligned.
	__shared__ alignas(16) int tile_shared[tile_size];

	// 1. a) Initialize shared memory barrier with the number of threads
	// participating in the barrier.
	//    b) Make initialized barrier visible in async proxy.
	__shared__ barrier bar;
	if (threadIdx.x == 0) {
		init(&bar, blockDim.x);				 // a)
		cde::fence_proxy_async_shared_cta(); // b)
	}
	__syncthreads();

	// 2. Initiate TMA transfer to copy global to shared memory.
	barrier::arrival_token token;
	if (threadIdx.x == 0) {
        printf("blockid:%d, blockDim.x * blockIdx.x:%d\n", blockIdx.x, blockDim.x * blockIdx.x);
        // coordinate表示偏移，coords 决定的偏移是 global memory 的偏移，不是 shared memory 的。
        // 如果是smem的offset，那么没道理了，一个smem的大小是16，这边host是16*3，也就是从16*3开始拷贝
        // 到当前的smem，然后去操作
        // 拷贝的大小由host端的box size决定的
		cde::cp_async_bulk_tensor_1d_global_to_shared(tile_shared, &tensor_map,
													  coordinate + blockDim.x * blockIdx.x, bar);
		// 第二个参数是，到达计数的更新值（通常是 1，表示一个线程已到达 barrier）。
		// 第三个参数是，以字节数为粒度的计数													  
		// 3a. Arrive on the barrier and tell how many bytes are expected to
		// come in (the transaction count)
		token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile_shared));
	} else {
		// 3b. Rest of threads just arrive
		token = bar.arrive();
	}

	// 3c. Wait for the data to have arrived.
	bar.wait(std::move(token));

	// 4. change the value in shared memory
	for (int i = threadIdx.x; i < array_size; i += blockDim.x) {
		if (i < tile_size) {
			tile_shared[i] += 1;
		}
	}

	// 5. Wait for shared memory writes to be visible to TMA engine.
	cde::fence_proxy_async_shared_cta();
	__syncthreads();
	// After syncthreads, writes by all threads are visible to TMA engine.

	// 6. Initiate TMA transfer to copy shared memory to global memory
	if (threadIdx.x == 0) {
		cde::cp_async_bulk_tensor_1d_shared_to_global(&tensor_map, coordinate + blockDim.x * blockIdx.x,
													  tile_shared);
		// 7. Wait for TMA transfer to have finished reading shared memory.
		// Create a "bulk async-group" out of the previous bulk copy operation.
		cde::cp_async_bulk_commit_group();
		// Wait for the group to have completed reading from shared memory.
		cde::cp_async_bulk_wait_group_read<0>();
	}

	__threadfence();
	__syncthreads();
}

// 原本的逻辑是只开一个block，然后只修改这一个block里面的第四行的16个数据
// 现在改成了开启四个block，每一个block的第四行都修改，然后打印
int main() {
	// initialize array and fill it with values
	int h_data[array_size];
	for (size_t i = 0; i < array_size; ++i) {
		h_data[i] = i;
	}

	// print the array before the kernel
	// one tile per line
	print_matrix(h_data, array_size / tile_size, tile_size);

	// transfer array to device
	int *d_data = nullptr;
	cudaMalloc(&d_data, array_size * sizeof(int));
	cudaMemcpy(d_data, h_data, array_size * sizeof(int),
			   cudaMemcpyHostToDevice);

    // array_size是需要拷贝的大小，tile_size是每次拷贝到smem的大小，d_data就是global memory的地址
	// create tensor map
	CUtensorMap tensor_map =
		create_1d_tensor_map(array_size, tile_size, d_data);

	size_t offset =
		tile_size * 3; // select the second tile of the array to change
	kernel<<<4, 128>>>(tensor_map, offset);

	cuda_check_error();

	cudaMemcpy(h_data, d_data, array_size * sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	// print the array after the kernel
	print_matrix(h_data, array_size / tile_size, tile_size);

	return 0;
}
