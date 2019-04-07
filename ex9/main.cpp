/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 09
 *
 *                           Group : TODO: TBD
 *
 *                            File : main.cpp
 *
 *                         Purpose : Stencil Code
 *
 *************************************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>
#include <cuda_runtime.h>

const static int DEFAULT_NUM_ELEMENTS   = 1024;
const static int DEFAULT_NUM_ITERATIONS =    5;
const static int DEFAULT_BLOCK_DIM      =  128;

//
// Structures
struct StencilArray_t {
    float* array;
    int    size; // size == width == height
};

//
// Function Prototypes
//
void printHelp(char *);

//
// Stencil Code Kernel for the speed calculation
//
extern void simpleStencil_Kernel_Wrapper(int gridSize, int blockSize, float* d_array, float* t_array, int size /* TODO Parameters */);
extern void optStencil_Kernel_Wrapper(int gridSize, int blockSize /* TODO Parameters */);

//
// Grid printing
//
void printgrid(float* grid, int size){
  for(int i =0; i<size*size; i++){
    if(i%size==0)
      std::cout << std::endl;
    std::cout << grid[i] << " ";
  }
  std::cout << std::endl;
}

//
// Main
//
int
main(int argc, char * argv[])
{
    bool showHelp = chCommandLineGetBool("h", argc, argv);
    if (!showHelp) {
        showHelp = chCommandLineGetBool("help", argc, argv);
    }

    if (showHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    //std::cout << "***" << std::endl
      //        << "*** Starting ..." << std::endl
        //      << "***" << std::endl;

    ChTimer memCpyH2DTimer, memCpyD2HTimer;
    ChTimer kernelTimer;

    //
    // Allocate Memory
    //
    int numElements = 0;
    chCommandLineGet<int>(&numElements, "s", argc, argv);
    chCommandLineGet<int>(&numElements, "size", argc, argv);
    numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;

    //
    // Host Memory
    //
    bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!pinnedMemory) {
        pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
    }

	// use opt kernel?
	bool useOpt = chCommandLineGetBool("opt", argc, argv);

    StencilArray_t h_array;
    h_array.size = numElements;
    if (!pinnedMemory) {
        // Pageable
        h_array.array = static_cast<float*>
                (malloc(static_cast<size_t>
                (h_array.size * sizeof(*(h_array.array)))));
    } else {
        // Pinned<F4>
        cudaMallocHost(&(h_array.array), 
                static_cast<size_t>
                (h_array.size * h_array.size * sizeof(*(h_array.array))));
    }

    // Init Particles
//  srand(static_cast<unsigned>(time(0)));
    srand(0); // Always the same random numbers
    for (int i = 0; i < h_array.size * h_array.size; i++) {
        h_array.array[i] = 0;
        if(i <= 3*h_array.size/4 && i >= h_array.size/4)
           h_array.array[i] = 127;

        // TODO: Initialize the array
    }
	std::cout << "size: " << h_array.size << "\n" << std::endl;
    printgrid(h_array.array, h_array.size);

    // Device Memory
    StencilArray_t d_array, t_array;
    d_array.size = t_array.size = h_array.size;
    cudaMalloc(&(d_array.array), 
            static_cast<size_t>(d_array.size * d_array.size * sizeof(*d_array.array)));
    cudaMalloc(&(t_array.array), 
            static_cast<size_t>(t_array.size * t_array.size * sizeof(*t_array.array)));

    if (h_array.array == NULL || d_array.array == NULL) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    //
    // Copy Data to the Device
    //
    memCpyH2DTimer.start();

    cudaMemcpy(d_array.array, h_array.array, 
            static_cast<size_t>(d_array.size * d_array.size * sizeof(*d_array.array)), 
            cudaMemcpyHostToDevice);

    memCpyH2DTimer.stop();

    cudaMemcpy(t_array.array, d_array.array, 
            static_cast<size_t>(d_array.size * d_array.size * sizeof(*d_array.array)), 
            cudaMemcpyDeviceToDevice);

    //
    // Get Kernel Launch Parameters
    //
    int blockSize = 0,
        gridSize = 0,
        numIterations = 0;

    // Number of Iterations 
    chCommandLineGet<int>(&numIterations,"i", argc, argv);
    chCommandLineGet<int>(&numIterations,"num-iterations", argc, argv);
    numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

    // Block Dimension / Threads per Block
    chCommandLineGet<int>(&blockSize,"t", argc, argv);
    chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
    blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

    if (blockSize > 1024) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - The number of threads per block is too big" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    gridSize = ceil(static_cast<float>(d_array.size) / static_cast<float>(blockSize));

    dim3 grid_dim = dim3(gridSize);
    dim3 block_dim = dim3(blockSize);

    //std::cout << "***" << std::endl;
    //std::cout << "*** Grid: " << gridSize << std::endl;
    //std::cout << "*** Block: " << blockSize << std::endl;
    //std::cout << "***" << std::endl;

    kernelTimer.start();
	
    for (int i = 0; i < numIterations; i ++) {
		if (!useOpt)
			simpleStencil_Kernel_Wrapper(gridSize, blockSize, d_array.array, t_array.array, d_array.size /* TODO Parameters */);
		else
			optStencil_Kernel_Wrapper(gridSize, blockSize /* TODO Parameters */);
    }

    // Synchronize
    cudaDeviceSynchronize();

    // Check for Errors
    cudaError_t cudaError = cudaGetLastError();
    if ( cudaError != cudaSuccess ) {
        std::cout << "\033[31m***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                  << std::endl
                  << "***\033[0m" << std::endl;

        return -1;
    }

    kernelTimer.stop();

    //
    // Copy Back Data
    //
    memCpyD2HTimer.start();

    cudaMemcpy(h_array.array, d_array.array, 
            static_cast<size_t>(h_array.size * h_array.size * sizeof(*(h_array.array))), 
            cudaMemcpyDeviceToHost);

    memCpyD2HTimer.stop();

    printgrid(h_array.array, h_array.size);

    // Free Memory
    if (!pinnedMemory) {
        free(h_array.array);
    } else {
        cudaFreeHost(h_array.array);
    }

    cudaFree(d_array.array);

    // Print Meassurement Results
    /*std::cout << "***" << std::endl
              << "*** Results:" << std::endl
              << "*** You made it. The GPU ran without an error!" << std::endl << std::endl
              << "***    Size: " << numElements << std::endl
              << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: " 
                << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(*h_array.array))
                << " GB/s" << std::endl
              << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: " 
                << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(*h_array.array))
                << " GB/s" << std::endl
              << "***    Time for Stencil Computation: " << 1e3 * kernelTimer.getTime()
                << " ms" << std::endl
              << "***" << std::endl;*/

    return 0;
}

void
printHelp(char * argv)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
                  << std::endl
              << "" << std::endl
              << "  -p|--pinned-memory" << std::endl
              << "    Use pinned Memory instead of pageable memory" << std::endl
              << "" << std::endl
              << "  -s <width-and-height>|--size <width-and-height>" << std::endl
              << "    THe width and the height of the array" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
                  << std::endl
              << "    The number of threads per block" << std::endl
              << "" << std::endl
              << "  --opt" 
                  << std::endl
              << "    Use the optimized Kernel" << std::endl
              << "" << std::endl;
}
