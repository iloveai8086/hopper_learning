sm_version=90a
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./headers/device/ -I./headers/host/
OPTIMIZATION=-O0
LINKS=-lcudart -lcuda
OUTPUT=bin

all:
	make test
	make run

test:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} example/test.cu

1:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/1_cluster.cu

2:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/2_wgmma_dense.cu

3:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/3_wgmma_sparse.cu

4:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/4_tma_1d.cu

5:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/5_tma_2d.cu

6:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/6_multicast.cu

7:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/7_reduce_store.cu
	
8:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/8_swizzle_manual.cu

9:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} examples/9_swizzle.cu

push:
	git add .
	git commit -m "update"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}