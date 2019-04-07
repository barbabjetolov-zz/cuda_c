.PHONY: build
build: ./bin/nbody

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/nbody: ./src/main.cu
	nvcc -O2 -arch=sm_20 --compiler-options "-O2 -Wextra" -o ./bin/nbody ./src/main.cu -I./inc/
