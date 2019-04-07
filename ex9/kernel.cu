 
/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 09
 *
 *                           Group : TODO
 *
 *                            File : kernel.cu
 *
 *                         Purpose : Stencil Code
 *
 *************************************************************************************************/


//
// Stencil Code Kernel for the heat calculation
//
__global__ void
simpleStencil_Kernel(float *d, float* t, int size/* TODO Parameters */)
{	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int num = size*size;

	//update the grid sites beside the borders
	if(i > size &&  
	   i % size != 0 &&
	   (i + 1) % size != 0 &&
	   i < num - size){
		
		t[i] = d[i] + 0.24 * (-4*d[i] + d[i+1] + d[i-1] + d[i+size] + d[i-size]);
		if(t[i] > 127)
			t[i] = 127;
	}

	__syncthreads();

	if(i == 0)
		for(int j=0; j<num; j++)
			d[j] = t[j];

}

void simpleStencil_Kernel_Wrapper(int gridSize, int blockSize, float* d_array, float* t_array, int size /* TODO Parameters */) {
	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);
	
	simpleStencil_Kernel<<<grid_dim, block_dim>>>(d_array, t_array, size /* TODO Parameters */);
}


//
// optimized Stencil Code Kernel for the heat calculation
//
__global__ void
optStencil_Kernel(/* TODO Parameters */)
{
	
}


void optStencil_Kernel_Wrapper(int gridSize, int blockSize /* TODO Parameters */) {
	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);
	
	optStencil_Kernel<<<grid_dim, block_dim >>>(/* TODO Parameters */);
}

