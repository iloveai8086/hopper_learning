// This code uses TMA's 2d load to load a matrix's tile to
// shared memory and then change the value in the
// shared memory and uses TMA's store to store the
// tile back to global memory. We print the result matrix to prove the
// changes are done

// note very carefully the order of the m and k coordinate in the api calls
// and note the alignment requirement of the coordinatess

#include <cuda.h>
#include <cuda/barrier>
#include <stdio.h>

#include "matrix_utilities.cuh"
#include "profile_utilities.cuh"
#include "tma.cuh"
#include "tma_tensor_map.cuh"

// Suppress warning about barrier in shared memory
#pragma nv_diag_suppress static_var_with_dynamic_init

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr size_t SMEM_HEIGHT = 16;
constexpr size_t SMEM_WIDTH = 8;
constexpr size_t M = 64; // Number of rows of matrix
constexpr size_t K = 32; // Number of columns of matrix
constexpr size_t gmem_len = M * K;

constexpr int m = 16; // subtile rows
constexpr int k = 8;  // subtile columns

static constexpr int buf_len = k * m;

__device__ static __forceinline__ void
init_barrier(uint64_t *bar, int thread_count, int transaction_count) {
	uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
	asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
				 "r"(thread_count + transaction_count));
}

__device__ static __forceinline__ void wait(uint64_t *bar, int kPhaseBit) {
	uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
	asm volatile("{\n"
				 ".reg .pred                P1;\n"
				 "LAB_WAIT:\n"
				 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
				 "@P1                       bra.uni DONE;\n"
				 "bra.uni                   LAB_WAIT;\n"
				 "DONE:\n"
				 "}\n" ::"r"(mbar_ptr),
				 "r"(kPhaseBit));
}

__device__ static __forceinline__ void arrive(uint64_t *bar,
											  uint32_t count = 1) {
	uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
	asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
				 :
				 : "r"(mbar_ptr), "r"(count)
				 : "memory");
}

__global__ void kernel2(const __grid_constant__ CUtensorMap tensor_map, int x,
						int y) {
	// 1) TMA bulk 需要 128B 对齐
	__shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

	// 2) 业务同步：两个 mbarrier（full：g2s 完成；empty：消费者完成）
	__shared__ uint64_t bar_full;
	__shared__ uint64_t bar_empty;
	__shared__ int phase_full;
	__shared__ int phase_empty;

	// 3) 仅供 TMA API 使用的 barrier（不参与业务同步）
	using barrier_t = cuda::barrier<cuda::thread_scope_block>;
	__shared__ barrier_t tma_bar;

	if (threadIdx.x == 0) {
		// 初始化给 TMA 用的 barrier（预期到达次数 = 全 block 线程数）
		init(&tma_bar, blockDim.x);

		// 代理可见性 fence（async -> generic）
		cde::fence_proxy_async_shared_cta();

		// 业务 mbarrier：只用 transaction_count
		init_barrier(&bar_full, 0, 1);	// 生产者 arrive 1 次
		init_barrier(&bar_empty, 0, 1); // 一个消费者代表 arrive 1 次
		phase_full = 0;
		phase_empty = 0;
	}
	__syncthreads();

	// ===== 使用 CUDA barrier 完整地把 g2s “做完并发布” =====
	barrier_t::arrival_token token;
	if (threadIdx.x == 0) {
		// 发起 g2s：注意传 smem_buffer（不是 &smem_buffer）
		cde::cp_async_bulk_tensor_2d_global_to_shared(smem_buffer, &tensor_map,
													  x, y, tma_bar);
		// 指定需等待的字节数（完整 tile）
		token =
			cuda::device::barrier_arrive_tx(tma_bar, 1, sizeof(smem_buffer));
	} else {
		// 其他线程参与到这个 barrier 的计数
		token = tma_bar.arrive();
	}

	// 所有线程都等到 TMA 发布完成（这一步对“稳定可见”非常关键）
	tma_bar.wait(cuda::std::move(token));
	__syncthreads();

	// 保险起见：再做一次 async -> generic 的可见性 fence
	cde::fence_proxy_async_shared_cta();

	// ===== 业务层：生产者通知“数据就绪”（mbarrier）=====
	if (threadIdx.x == 0) {
		arrive(&bar_full, 1);
	}

	// ===== 消费者：等待数据可读（mbarrier）=====
	wait(&bar_full, phase_full);
	if (threadIdx.x == 0)
		phase_full ^= 1;
	__syncthreads();

	// ===== 消费者更新（按你当前逻辑，只改第 0 行）=====
	// for (int i = threadIdx.x; i < (int)SMEM_WIDTH; i += blockDim.x) {
	// 	smem_buffer[0][i] += threadIdx.x;
	// }
	smem_buffer[0][threadIdx.x] += threadIdx.x;

	// ===== 消费者完成：一个代表通知（mbarrier）=====
	__syncthreads(); // 收拢所有消费者（代表制）
	if (threadIdx.x == 1) {
		arrive(&bar_empty,
			   1); // 代表 arrive（若改为 N 个消费者各自 arrive，则把 tc=N
				   // 并去掉这句上面的 __syncthreads()）
	}

	// ===== 生产者等待消费者完成并写回 =====
	if (threadIdx.x == 0) {
		wait(&bar_empty, phase_empty);
		phase_empty ^= 1;

		// generic -> async 的可见性 fence（让 async 代理能看到消费者写入的
		// smem）
		cde::fence_proxy_async_shared_cta();

		// 回写：注意传 smem_buffer（不是 &smem_buffer）
		cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y,
													  smem_buffer);
		cde::cp_async_bulk_commit_group();
		cde::cp_async_bulk_wait_group_read<0>();
	}
}

#define TILE_BYTES (SMEM_HEIGHT * SMEM_WIDTH * sizeof(int))
#define STR1(x) #x
#define STR(x) STR1(x)

// 仅此 kernel2 改为纯 mbarrier + PTX TMA，完全无 cuda::barrier 和 token
__global__ void kernel3(const __grid_constant__ CUtensorMap tensor_map, int x,
						int y) {
	__shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

	// mbarrier（纯 PTX）：full = g2s 完成；empty = 消费者完成
	__shared__ uint64_t bar_full;
	__shared__ uint64_t bar_empty;
	__shared__ int phase_full;
	__shared__ int phase_empty;

	if (threadIdx.x == 0) {
		// 只用 transaction_count
		init_barrier(&bar_full, 0, 1); // g2s 完成时由 TMA 自动 arrive 一次
		init_barrier(
			&bar_empty, 0,
			1); // 代表 arrive 一次（如需 N 个消费者各自 arrive，则把 tc=N）
		phase_full = 0;
		phase_empty = 0;
	}
	__syncthreads();

	// ===== 生产者：TMA g2s（PTX）把 complete_tx 绑定 bar_full =====
	if (threadIdx.x == 0) {
		// 用 64-bit 地址：shared 目标地址、mbarrier 地址、tensor map 指针
		unsigned long long smem_ptr = __cvta_generic_to_shared(smem_buffer);
		unsigned long long mbar_ptr = __cvta_generic_to_shared(&bar_full);
		unsigned long long tmap_ptr = (unsigned long long)(&tensor_map);
	  
		// TODO 这边后面改一下，调用不对
		// asm volatile(
		// 	"{\n\t"
		// 	".reg .b64 smp, tmap, mbar;\n\t"
		// 	"mov.b64 smp,  %0;\n\t"     // smem 目标（shared addr，经 cvta 转换）
		// 	"mov.b64 tmap, %1;\n\t"     // CUtensorMap*（64-bit 通用/常量/参数指针）
		// 	"mov.b64 mbar, %4;\n\t"     // mbarrier 在 shared 的地址（64-bit）
		// 	// g2s + 绑定 mbarrier 完成事务；bytes 必须是字面量
		// 	"cp.async.bulk.tensor.2d.shared::cta.global"
		// 	".mbarrier::complete_tx::bytes "
		// 	"[smp], [tmap, {%2, %3}], [mbar], " STR(TILE_BYTES) ";\n\t"
		// 	"}\n"
		// 	:
		// 	: "l"(smem_ptr),   // %0
		// 	  "l"(tmap_ptr),   // %1
		// 	  "r"(x),          // %2
		// 	  "r"(y),          // %3
		// 	  "l"(mbar_ptr)    // %4
		// 	: "memory");
	  
		// 提交/读侧组就绪；complete_tx 会对 bar_full 执行 arrive（无需你手动 arrive）
		cde::cp_async_bulk_commit_group();
		cde::cp_async_bulk_wait_group_read<0>();
	  }

	// ===== 消费者：等待 g2s 完成（纯 mbarrier，无 token）=====
	wait(&bar_full, phase_full);
	if (threadIdx.x == 0)
		phase_full ^= 1;
	__syncthreads(); // 统一开始读/改 smem 的时点，方便可见性推理

	// ===== 消费者修改（只改第 0 行第 threadIdx.x 列）=====
	if (threadIdx.x < SMEM_WIDTH) {
		smem_buffer[0][threadIdx.x] += threadIdx.x;
	}

	// ===== 消费者完成：代表 arrive（或改为 N 个消费者各自 arrive）=====
	__syncthreads(); // 代表制下收拢所有消费者
	if (threadIdx.x == 1) {
		arrive(&bar_empty, 1);
	}

	// ===== 生产者：等待消费者完成并写回 =====
	if (threadIdx.x == 0) {
		wait(&bar_empty, phase_empty);
		phase_empty ^= 1;

		// generic -> async：把消费者的 smem 写发布给 TMA store
		cuda::device::experimental::fence_proxy_async_shared_cta();

		// s2g（API 这边不需要 barrier）；注意传 smem_buffer（不要 &）
		cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(
			&tensor_map, x, y, smem_buffer);
		cuda::device::experimental::cp_async_bulk_commit_group();
		cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
	}
}

int main() {
	// fill the host matrix
	int host_tensor[gmem_len]; // gmem_len = M * K = 64 * 32 = 2048
	// 按照smem为一个tile填满tensor，所填的tensor为tile id
	fill_tilewise(host_tensor, M, K, m, k);

	print_matrix(host_tensor, M, K);

	// copy host matrix to device
	int *tensor_ptr = nullptr;
	cudaMalloc(&tensor_ptr, gmem_len * sizeof(int));
	cudaMemcpy(tensor_ptr, host_tensor, gmem_len * sizeof(int),
			   cudaMemcpyHostToDevice);

	// create tensor map for the matrix
	CUtensorMap tensor_map = create_2d_tensor_map(M, K, m, k, tensor_ptr);

	// launch kernel, select a tile coordinate
	// x (0 16 32 48) y (0 8 16 24) must be aligned with m(16) and k(8)
	int coordinate_m = 48; // 48 / 16 = 3
	int coordinate_k = 24; // 24 / 8 = 3
	// test<<<1, 128>>>(tensor_map, coordinate_k, coordinate_m);
	// kernel2<<<1, 128>>>(tensor_map, coordinate_k, coordinate_m);
	kernel3<<<1, 128>>>(tensor_map, coordinate_k, coordinate_m);

	cuda_check_error();

	// copy device matrix to host
	int host_gmem_tensor[gmem_len];
	cudaMemcpy(host_gmem_tensor, tensor_ptr, gmem_len * sizeof(int),
			   cudaMemcpyDeviceToHost);

	// verify the results
	print_matrix(host_gmem_tensor, M, K);

	return 0;
}
