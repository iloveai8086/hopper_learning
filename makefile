sm_version=90
NVCC=/usr/local/cuda-13.0/bin/nvcc
INCLUDES=-I./headers/device/ -I./headers/host/
OPTIMIZATION=-O0 -lineinfo
LINKS=-lcudart -lcuda
OUTPUT=run
# KERNEL=examples/6_multicast.cu
# KERNEL=examples/1_cluster.cu
# KERNEL=examples/1_2_cluter_dsm.cu
# KERNEL=examples/4_tma_1d_v2.cu
# KERNEL=examples/4_tma_1d.cu
# KERNEL=examples/5_tma_2d.cu
KERNEL=examples/5_tma_2d_v2.cu
COMMENT=update

all:
	make kernel
	make run

kernel:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} ${KERNEL}

push:
	git add .
	git commit -m "${COMMENT}"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}
