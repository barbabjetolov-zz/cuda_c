/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : TODO
 *
 *                   File : main.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

const static int DEFAULT_NUM_ELEMENTS   = 1024;
const static int DEFAULT_NUM_ITERATIONS =    5;
const static int DEFAULT_BLOCK_DIM      =  128;

const static float TIMESTEP =      1e-6; // s
const static float GAMMA    = 6.673e-11; // (Nm^2)/(kg^2)
//const static float EPS    = 1e-8;


//
// Structures
//
// Use a SOA (Structure of Arrays)
//
struct Body_t {
	float4* posMass;  /* x = x */
	                  /* y = y */
	                  /* z = z */
	                  /* w = Mass */
	float3* velocity; /* x = v_x*/
	                  /* y = v_y */
	                  /* z= v_z */

	Body_t(): posMass(NULL), velocity(NULL) {}
	};

//
// Function Prototypes
//
void printHelp(char *);
void printElement(Body_t, int, int);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
	float squaredistance  = ((a.x - b.x)*(a.x - b.x) +
	 		 							(a.y - b.y)*(a.y - b.y) +
			 							(a.z - b.z)*(a.z - b.z));
	return squaredistance;
}

//
// Calculate the forces between two bodies
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3& force)
{
	float temp = getDistance(bodyA, bodyB);
	temp = bodyB.w/__powf(temp, 1.5);

	force.x += temp * (bodyA.x - bodyB.x);
	force.y += temp * (bodyA.y - bodyB.y);
	force.z += temp * (bodyA.z - bodyB.z);
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3& currentSpeed, float3 force)
{
	currentSpeed.x += -GAMMA * force.x * TIMESTEP;
	currentSpeed.y += -GAMMA * force.y * TIMESTEP;
	currentSpeed.z += -GAMMA * force.z * TIMESTEP;
}

//
// n-Body Kernel for the speed calculation
//
__global__ void
simpleNbody_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;

	if (elementId < numElements) {
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		elementForce = make_float3(0,0,0);

		for (int i = 0; i < numElements; i++) {
			if (i != elementId) {
				bodyBodyInteraction(elementPosMass, bodyPos[i], elementForce);
			}
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		bodySpeed[elementId] = elementSpeed;
	}
}

//
// n-Body Kernel for Energy calculation
//
float Energy_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{

	float3 elementSpeed;
	float distance;
	float Energy = 0;

	for (int j = 0; j < numElements; j++){

		elementSpeed = bodySpeed[j];
		Energy += 0.5 * bodyPos[j].w * elementSpeed.x * elementSpeed.x;
		Energy += 0.5 * bodyPos[j].w * elementSpeed.y * elementSpeed.y;
		Energy += 0.5 * bodyPos[j].w * elementSpeed.z * elementSpeed.z;

		for (int i = 0; i < numElements; i++) {
			if (i != j) {
				distance = ((bodyPos[j].x - bodyPos[i].x)*(bodyPos[j].x - bodyPos[i].x) +
				 		 	(bodyPos[j].y - bodyPos[i].y)*(bodyPos[j].y - bodyPos[i].y) +
						 	(bodyPos[j].z - bodyPos[i].z)*(bodyPos[j].z - bodyPos[i].z));

				Energy += (- GAMMA * bodyPos[j].w * bodyPos[i].w)/((distance)*(distance));
			}
		}
	}
	return Energy;
}

//
// n-Body Kernel to update the position
// Neended to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, float4* bodyPos, float3* bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementSpeed;

	if (elementId < numElements) {
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];

		elementPosMass.x += elementSpeed.x * TIMESTEP;
		elementPosMass.y += elementSpeed.y * TIMESTEP;
		elementPosMass.z += elementSpeed.z * TIMESTEP;

		bodyPos[elementId] = elementPosMass;
	}
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

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ?
			numElements : DEFAULT_NUM_ELEMENTS;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory) {
		pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
	}

	Body_t h_particles;
	if (!pinnedMemory) {
		// Pageable
		h_particles.posMass = static_cast<float4*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass)))));
		h_particles.velocity = static_cast<float3*>
				(malloc(static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity)))));
	} else {
		// Pinned
		cudaMallocHost(&(h_particles.posMass),
				static_cast<size_t>
				(numElements * sizeof(*(h_particles.posMass))));
		cudaMallocHost(&(h_particles.velocity),
				static_cast<size_t>
				(numElements * sizeof(*(h_particles.velocity))));
	}

	// Init Particles
//	srand(static_cast<unsigned>(time(0)));
	srand(0); // Always the same random numbers
	for (int i = 0; i < numElements; i++) {
		h_particles.posMass[i].x = 1e-8*static_cast<float>(rand()); // Modify the random values to
		h_particles.posMass[i].y = 1e-8*static_cast<float>(rand()); // increase the position changes
		h_particles.posMass[i].z = 1e-8*static_cast<float>(rand()); // and the velocity
		h_particles.posMass[i].w =  1e4*static_cast<float>(rand());
		h_particles.velocity[i].x = 0.0f;
		h_particles.velocity[i].y = 0.0f;
		h_particles.velocity[i].z = 0.0f;
	}

	printElement(h_particles, 0, 0);
	// Device Memory
	Body_t d_particles;
	cudaMalloc(&(d_particles.posMass),
			static_cast<size_t>(numElements * sizeof(*(d_particles.posMass))));
	cudaMalloc(&(d_particles.velocity),
			static_cast<size_t>(numElements * sizeof(*(d_particles.velocity))));

	if (h_particles.posMass == NULL || h_particles.velocity == NULL ||
		d_particles.posMass == NULL || d_particles.velocity == NULL) {
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_particles.posMass, h_particles.posMass,
			static_cast<size_t>(numElements * sizeof(float4)),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_particles.velocity, h_particles.velocity,
			static_cast<size_t>(numElements * sizeof(float3)),
			cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

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

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;


	// check energy (on CPU) -c
	bool check_energy = chCommandLineGetBool("c", argc, argv);
	// no intermediate print -np
	bool no_print = chCommandLineGetBool("np", argc, argv);

	if (check_energy) {
		float E_i = Energy_Kernel(numElements, h_particles.posMass, h_particles.velocity);
		std::cout << "***" << std::endl
						<< "*** Check initial energy : "<< E_i << std::endl
						<< "***"<< std::endl;
					}

	kernelTimer.start();
	for (int i = 0; i < numIterations; i ++) {

		simpleNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
				d_particles.velocity);
		
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
				d_particles.velocity);

		cudaMemcpy(h_particles.posMass, d_particles.posMass, sizeof(float4), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_particles.velocity, d_particles.velocity, sizeof(float3), cudaMemcpyDeviceToHost);
		if (!no_print){
			printElement(h_particles, 0, i+1);
		  }
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if ( cudaError != cudaSuccess ) {
		std::cout << "***" << std::endl
		          << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
		          << std::endl
		          << "***" << std::endl;

		return -1;
	}

	kernelTimer.stop();

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_particles.posMass, d_particles.posMass,
			static_cast<size_t>(numElements * sizeof(*(h_particles.posMass))),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particles.velocity, d_particles.velocity,
			static_cast<size_t>(numElements * sizeof(*(h_particles.velocity))),
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	if (check_energy) {
		float E_f = Energy_Kernel(numElements, h_particles.posMass, h_particles.velocity);
		std::cout << "***" << std::endl
						<< "*** Check final energy : "<< E_f << std::endl
						<< "***" << std::endl;
					}
	if (no_print){
			printElement(h_particles, 0, numIterations - 1);
	 }


	// Free Memory
	if (!pinnedMemory) {
		free(h_particles.posMass);
		free(h_particles.velocity);
	} else {
		cudaFreeHost(h_particles.posMass);
		cudaFreeHost(h_particles.velocity);
	}

	cudaFree(d_particles.posMass);
	cudaFree(d_particles.velocity);

	// Print Meassurement Results
    std::cout << "***" << std::endl
              << "*** Results:" << std::endl
              << "***    Num Elements: " << numElements << std::endl
              << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: "
                << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
                << " GB/s" << std::endl
              << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
                << " ms" << std::endl
              << "***    Copy Bandwidth: "
                << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
                << " GB/s" << std::endl
              << "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
                << " ms" << std::endl
              << "***" << std::endl;

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
              << "  -s <num-elements>|--size <num-elements>" << std::endl
              << "    Number of elements (particles)" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
                  << std::endl
              << "    The number of threads per block" << std::endl
              << "" << std::endl;
}

//
// Print one element
//
void
printElement(Body_t particles, int elementId, int iteration)
{
    float4 posMass = particles.posMass[elementId];
    float3 velocity = particles.velocity[elementId];

    std::cout << "***" << std::endl
              << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
              << "***" << std::endl
              << "*** Position: <"
                  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
                  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
                  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
              << "*** velocity: <"
                  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
                  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
                  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
              << "*** Mass: "
                  << std::setw(11) << std::setprecision(9) << posMass.w << " kg"<< std::endl
              << "***" << std::endl;
}
