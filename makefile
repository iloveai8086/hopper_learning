sm_version=90a
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=-I./headers/device/ -I./headers/host/
OPTIMIZATION=-O0
LINKS=-lcudart -lcuda
OUTPUT=bin
KERNEL=1_n8.cu

all:
	make kernel
	make run

kernel:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} ${KERNEL}

push:
	git add .
	git commit -m "update"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}