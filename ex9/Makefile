NVCC=nvcc
INC=-I./inc -I/usr/local/cuda/include
LIB=-L/usr/local/cuda/lib64

.PHONY: build
build: ./bin/stencil 

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/stencil: ./src/main.cu 
	$(NVCC) -O2 -arch=sm_20 --compiler-options "-O2 -Wextra" -o $@ $^ $(INC) $(LIB)
