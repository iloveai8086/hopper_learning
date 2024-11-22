// This code uses TMA's 2d load to load a matrix's tile to
// shared memory and then change the value in the
// shared memory and uses TMA's store to store the
// tile back to global memory. We print the result matrix to prove the
// changes are done

// note very carefully the order of the m and k coordinate in the api calls
// and note the alignment requirement of the coordinatess

#include <cuda/barrier>
#include <stdio.h>
#include <cuda.h>

#include "tma_tensor_map.cuh"
#include "matrix_utilities.cuh"
#include "tma.cuh"
#include "profile_utilities.cuh"

// Suppress warning about barrier in shared memory
#pragma nv_diag_suppress static_var_with_dynamic_init

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr size_t M = 64; // Number of rows of matrix
constexpr size_t K = 16; // Number of columns of matrix
constexpr size_t gmem_len = M * K;

constexpr int m = 64; // subtile rows
constexpr int k = 16;  // subtile columns

static constexpr int buf_len = k * m;

__global__ void test(const __grid_constant__ CUtensorMap tensor_map, int x, int y)
{
  __shared__ alignas(128) half smem_buffer[buf_len];
  __shared__ barrier bar;

  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  // Load data:
  uint64_t token;
  if (threadIdx.x == 0)
  {
    // call the loading api
    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_buffer, &tensor_map, x, y, bar);
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  }
  else
  {
    token = bar.arrive();
  }

  bar.wait(cuda::std::move(token));

  __syncthreads();
  
  // print the matrix
 //  if (threadIdx.x == 0) {
 //    for (int r = 0; r < m; r++) {
	//   for (int c = 0; c < k; c++) {
	// 	printf("%d ", smem_buffer[r * k + c]);
	//   }
	//   printf("\n");
	// }
 //  }

  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // Write back to global memory:
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  __threadfence();
  __syncthreads();
}

int main()
{
  // fill the host matrix
  half host_tensor[gmem_len];
  fill_tilewise(host_tensor, M, K, 8, 8);

  // print_matrix(host_tensor, M, K);

  // copy host matrix to device
  int *tensor_ptr = nullptr;
  cudaMalloc(&tensor_ptr, gmem_len * sizeof(half));
  cudaMemcpy(tensor_ptr, host_tensor, gmem_len * sizeof(half), cudaMemcpyHostToDevice);

  // create tensor map for the matrix
  CUtensorMap tensor_map = create_2d_tensor_map_half(M, K, m, k, tensor_ptr);

  // launch kernel, select a tile coordinate
  // x (0 16 32 48) y (0 8 16 24) must be aligned with m and k
  int coordinate_m = 0;
  int coordinate_k = 0;
  test<<<1, 128>>>(tensor_map, coordinate_k, coordinate_m);

  cuda_check_error();

  // copy device matrix to host
  half host_gmem_tensor[gmem_len];
  cudaMemcpy(host_gmem_tensor, tensor_ptr, gmem_len * sizeof(half), cudaMemcpyDeviceToHost);

  // verify the results
  print_matrix(host_gmem_tensor, M, K);

  return 0;
}
