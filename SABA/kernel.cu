#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>  
#include <time.h>
#include <inttypes.h>

#include <stdlib.h>
#include <windows.h>
#include <stdio.h>
#include <algorithm>
#include <string>
#include <memory>
#include <sstream>
#include <errno.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif

using namespace std;

// * * * Experiment configuration
#define N 16
#define BLOCKS 1
#define START_ENERGY_NUMBER 8
#define END_ENERGY_NUMBER	9
#define EXPERIMENTS_PER_ENERGY_VALUE 3
#define STEP 1000

// make them global for deleting in upper function call
double *dev_X0			= nullptr; 
double *dev_TempPoint	= nullptr;
double *dev_EPS			= nullptr;
double *dev_Energy		= nullptr;
double *dev_Vars		= nullptr;
double *dev_Measures	= nullptr;
double *dev_Counters	= nullptr;

cudaError_t TakeMeAnswer(double *X0, double *EPS, double *Measures, int64_t M_size, double *Vars, char* fileNameM, char* fileNameT, char* fileNameL);

// * * * Uncomment if you want to collect measures
//#define MEASURES

// * * * Choose forward or backward run ( dont' forget that backward can be run only using forward results)
#define FWRD
//#define BKWD

// * * * In this mode only one central oscillator oscillates, use it for checking correctness of integrator
#define LINEAR

// Some utility staff
string toString(const int64_t& value)
{
	ostringstream oss;
	oss << value;
	string result = oss.str();
	return result;
}

__device__ void swapPointers(double **A, double **B)
{
	double *temp = *A;
	*A = *B;
	*B = temp;
}

// START: Main integrator logic
__device__ void LA(double tau, int64_t ID, double *X0, double *TempPoint, double *EPS)
{

	#ifdef LINEAR
	TempPoint[ID * 2] = X0[ID * 2] + tau * X0[ID * 2 + 1];
	TempPoint[ID * 2 + 1] = X0[ID * 2 + 1];
	#else
	TempPoint[ID * 2] = X0[ID * 2] + tau * X0[ID * 2 + 1];
	TempPoint[ID * 2 + 1] = X0[ID * 2 + 1];
	#endif
}

__device__ void LB(double tau, int64_t ID, double *X0, double *TempPoint, double *EPS)
{
	double ulM1_ul_3 = 0, ulP1_ul_3 = 0, Ql;

	// border conditions
	if (ID != 0)
	{
		ulM1_ul_3 = (X0[(ID - 1) * 2] - X0[ID * 2])*(X0[(ID - 1) * 2] - X0[ID * 2])*(X0[(ID - 1) * 2] - X0[ID * 2]);
	}
	else
	{
		ulM1_ul_3 = 0;
	}

	if (ID != N - 1)
	{
		ulP1_ul_3 = (X0[(ID + 1) * 2] - X0[ID * 2])*(X0[(ID + 1) * 2] - X0[ID * 2])*(X0[(ID + 1) * 2] - X0[ID * 2]);
	}
	else
	{
		ulP1_ul_3 = 0;
	}

	#ifdef LINEAR
	Ql = EPS[ID] * X0[ID * 2];
	TempPoint[ID * 2] = X0[ID * 2];
	TempPoint[ID * 2 + 1] = X0[ID * 2 + 1] - tau * Ql;
	#else
	Ql = EPS[ID] * X0[ID * 2] - ulM1_ul_3 - ulP1_ul_3;
	TempPoint[ID * 2] = X0[ID * 2];
	TempPoint[ID * 2 + 1] = X0[ID * 2 + 1] - tau * Ql;
	#endif
}

__device__ void LC(double tau, int64_t ID, double *X0, double *TempPoint, double *EPS)
{
	double  ul_minus_1, el_minus_1, ul_minus_2, ulM1_ul_2, ulM1_ul_3, ulM2_ulM1_3;
	double  ul_plus_1, el_plus_1, ul_plus_2, ulP1_ul_2, ulP1_ul_3, ulP2_ulP1_3;
	double  ul, pl, el, Ql, Ql_minus_1, Ql_plus_1;

	ul = X0[ID * 2];
	pl = X0[ID * 2 + 1];
	el = EPS[ID];

	// border conditions
	if (ID == 0) {
		ul_minus_1 = 0;
		el_minus_1 = 0;
		ul_minus_2 = 0;
		ulM1_ul_2 = 0;
		ulM1_ul_3 = 0;
		ulM2_ulM1_3 = 0;
	}
	else if (ID == 1){
		ul_minus_1 = X0[0];
		el_minus_1 = EPS[0];
		ul_minus_2 = 0;
		ulM1_ul_2 = (ul_minus_1 - ul)*(ul_minus_1 - ul);
		ulM1_ul_3 = (ul_minus_1 - ul)*ulM1_ul_2;
		ulM2_ulM1_3 = 0;
	}
	else {
		ul_minus_1 = X0[(ID - 1) * 2];
		el_minus_1 = EPS[ID - 1];
		ul_minus_2 = X0[(ID - 2) * 2];
		ulM1_ul_2 = (ul_minus_1 - ul)*(ul_minus_1 - ul);
		ulM1_ul_3 = (ul_minus_1 - ul)*ulM1_ul_2;
		ulM2_ulM1_3 = (ul_minus_2 - ul_minus_1)*(ul_minus_2 - ul_minus_1)*(ul_minus_2 - ul_minus_1);
	}

	if (ID == (N - 1)) {
		ul_plus_1 = 0;
		el_plus_1 = 0;
		ul_plus_2 = 0;
		ulP1_ul_2 = 0;
		ulP1_ul_3 = 0;
		ulP2_ulP1_3 = 0;
	}
	else if (ID == (N - 2)) {
		ul_plus_1 = X0[(ID + 1) * 2];
		el_plus_1 = EPS[ID + 1];
		ul_plus_2 = 0;
		ulP1_ul_2 = (ul_plus_1 - ul)*(ul_plus_1 - ul);
		ulP1_ul_3 = (ul_plus_1 - ul)*ulP1_ul_2;
		ulP2_ulP1_3 = 0;
	}
	else {
		ul_plus_1 = X0[(ID + 1) * 2];
		el_plus_1 = EPS[ID + 1];
		ul_plus_2 = X0[(ID + 2) * 2];
		ulP1_ul_2 = (ul_plus_1 - ul)*(ul_plus_1 - ul);
		ulP1_ul_3 = (ul_plus_1 - ul)*ulP1_ul_2;
		ulP2_ulP1_3 = (ul_plus_2 - ul_plus_1)*(ul_plus_2 - ul_plus_1)*(ul_plus_2 - ul_plus_1);
	}


	#ifdef LINEAR
	Ql = el*ul;
	TempPoint[ID * 2] = ul;
	TempPoint[ID * 2 + 1] = pl - 2 * tau * (el * Ql);
	#else
	Ql = el*ul - ulM1_ul_3 - ulP1_ul_3;
	Ql_minus_1 = el_minus_1*ul_minus_1 - ulM2_ulM1_3 - (-1)*ulM1_ul_3;
	Ql_plus_1 = el_plus_1*ul_plus_1 - (-1)*ulP1_ul_3 - ulP2_ulP1_3;

	TempPoint[ID * 2] = ul;
	TempPoint[ID * 2 + 1] = pl - 2 * tau * ((el + 3 * ulM1_ul_2 + 3 * ulP1_ul_2) * Ql -
		3 * Ql_minus_1 * ulM1_ul_2 - 3 * Ql_plus_1  * ulP1_ul_2);
	#endif
}

__device__ void CalcCharacteristics(int64_t ID, double *X0, double *EPS, double *Energy, double *Measures, int64_t CalcNum, double time, double E0)
{
	double ul, pl, el, ulP1_ul_4 = 0, ulM1_ul_4 = 0;

	ul = X0[ID * 2];
	pl = X0[ID * 2 + 1];
	el = EPS[ID];
	#ifndef LINEAR
	if (ID != N - 1) ulP1_ul_4 = (X0[(ID + 1) * 2] - ul)*(X0[(ID + 1) * 2] - ul)*(X0[(ID + 1) * 2] - ul)*(X0[(ID + 1) * 2] - ul);
	else ulP1_ul_4 = 0;

	if (ID != 0)   ulM1_ul_4 = (X0[(ID - 1) * 2] - ul)*(X0[(ID - 1) * 2] - ul)*(X0[(ID - 1) * 2] - ul)*(X0[(ID - 1) * 2] - ul);
	else ulM1_ul_4 = 0;
	#endif
	Energy[ID] = pl * pl *      0.5 +
		el * ul * ul * 0.5 +
		(1 / 8.0) * ulM1_ul_4 +
		(1 / 8.0) * ulP1_ul_4;
	__syncthreads();

	if (ID == 0)
	{
		double exp_energy = 0;
		for (int64_t i = 0; i<N; i++)
		{
			exp_energy += Energy[i];
		}

		double m1 = 0, m2 = 0;
		for (int64_t i = 0; i < N; i++) m1 += i * Energy[i];
		m1 = m1 / exp_energy;
		for (int64_t i = 0; i < N; i++) m2 += (i - m1) * (i - m1) * Energy[i];
		m2 = m2 / exp_energy;
		double P = 0;
		for (int64_t i = 0; i < N; i++) P += Energy[i] * Energy[i];
		P = P / exp_energy / exp_energy;
		P = 1 / P;
		double compactness_index = P * P / m2;
		
		// save results
		Measures[8 * CalcNum + 0] = time;													// time
		Measures[8 * CalcNum + 1] = m1;														// first momenta
		Measures[8 * CalcNum + 2] = m2;														// second momenta
		Measures[8 * CalcNum + 3] = P;														// participation number
		Measures[8 * CalcNum + 4] = compactness_index;										// compactness index
		Measures[8 * CalcNum + 5] = Energy[N / 2 - 1] + Energy[N / 2] + Energy[N / 2 + 1];	// summary energy of three central oscillators
		Measures[8 * CalcNum + 6] = exp_energy;												// current full system energy
		Measures[8 * CalcNum + 7] = log10(abs(exp_energy - E0));							// energy error
	}
}

__global__ void SABAKernel(double *gl_X0, double *gl_TempPoint, double *gl_EPS, double *gl_Energy, double *gl_Measures, double *gl_Vars, double *gl_counters)
{
	extern __shared__ double sh_allShData[];
	double *sh_TempPoint	= sh_allShData;
	double *sh_X0			= (double*)&sh_TempPoint[2 * N];
	double *sh_EPS			= (double*)&sh_X0[2 * N];
	double *sh_Vars			= (double*)&sh_EPS[N];
	double *sh_Energy		= (double*)&sh_Vars[12];
	double *sh_counters		= (double*)&sh_Energy[N];

	int64_t ID = threadIdx.x;
	int64_t BlockID = blockIdx.x;

	// threads copy values from global to shared memory
	sh_X0[ID * 2]		= gl_X0[ID * 2 + BlockID * 2 * N];
	sh_X0[ID * 2 + 1]	= gl_X0[ID * 2 + 1 + BlockID * 2 * N];
	sh_EPS[ID]			= gl_EPS[ID + BlockID*N];

	// 0 thread copy 10 values by itself
	if (ID == 0)
	{
		for (int64_t i = 0; i < 12; i++)
		{
			sh_Vars[i] = gl_Vars[i];
		}
		for (int64_t i = 0; i < 4; i++)
		{
			sh_counters[i] = gl_counters[i];
		}
	}

	// check all needed global memory has copied to shared memory, syncronize everyone
	__syncthreads();

#ifdef FWRD
	int64_t ALL_steps = round((sh_Vars[1] - sh_Vars[0]) / sh_Vars[3]);
	if (ID == 0) printf("ALL_steps = %i\n", ALL_steps);
	for (int64_t iter = 0; iter < ALL_steps; iter++)
	
#elif BKWD
	int64_t ALL_steps = round((sh_Vars[1] - sh_Vars[0]) / sh_Vars[3])+1;
	if (ID == 0) printf("ALL_steps = %i\n", ALL_steps);
	__syncthreads();
	for (int64_t iter = 0; iter < ALL_steps; iter++)	
#endif
	{
		LC(sh_Vars[7], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

		LA(sh_Vars[4], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

		LB(sh_Vars[6], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

		LA(sh_Vars[5], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

		LB(sh_Vars[6], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

		LA(sh_Vars[4], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();
		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();
		
		LC(sh_Vars[7], ID, sh_X0, sh_TempPoint, sh_EPS);
		__syncthreads();

 		swapPointers(&sh_TempPoint, &sh_X0);
		__syncthreads();

#ifdef MEASURES
		if (ID == 0)
		{
			if (sh_counters[0] > sh_counters[1]) 
			{
				sh_counters[2]++;//calcNUM
				sh_counters[0] = 0; //counter
				sh_counters[1] *= sh_Vars[11];//counterMax
				sh_counters[3] = 1;//flag//start calc
			}
			else sh_counters[0]++;
		}
		if (sh_counters[3] == 1 || (sh_Vars[1] == sh_Vars[10] && iter == ALL_steps -1))// t1 == sh_Vars[10])
		{
			if (sh_Vars[1] == sh_Vars[10] && iter == ALL_steps - 1)
				sh_counters[2]++;				
			double time_for_func = sh_Vars[0] + iter * sh_Vars[3];
			CalcCharacteristics(ID, sh_X0, sh_EPS, sh_Energy, &gl_Measures[BlockID * 8 * static_cast<int64_t>(gl_Vars[9])], (int64_t)(sh_counters[2]), time_for_func, sh_Vars[8]);
		}
		__syncthreads();
		if (ID == 0)
			sh_counters[3] = 0;
#endif
	}

	gl_X0[ID * 2 + BlockID * 2 * N] = sh_X0[ID * 2];
	gl_X0[ID * 2 + 1 + BlockID * 2 * N] = sh_X0[ID * 2 + 1];
	if (ID == 0)
	{
		gl_counters[0] = sh_counters[0];
		gl_counters[1] = sh_counters[1];
		gl_counters[2] = sh_counters[2];
		gl_counters[3] = sh_counters[3];
	}
}
// END: Main integrator logic

// START: Initialization
void InitPoint(double *X0, double E0)
{
	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
	{
		for (int64_t i = 0; i < N; i++)
		{
			X0[i * 2 + block_numb * 2 * N] = 0;
			X0[i * 2 + 1 + block_numb * 2 * N] = 0;
		}
		if (N % 2 == 0)
		{
			X0[N + 1 + block_numb * 2 * N] = pow((double)(2 * E0), (double)(0.5));
		}
		else
		{
			X0[N + block_numb * 2 * N] = pow((double)(2 * E0), (double)(0.5));
		}
	}
}

void InitPotential(double *EPS, double _minimum, double _maximum)
{
	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)	
	{
		for (int64_t i = 0; i < N; i++)
		{
			EPS[i + block_numb*N] = _minimum + double(rand()) / RAND_MAX * (_maximum - _minimum);
		}
	}
}

void InitPoint_BACKWARD(double *X0, string exp)
{
	FILE *LastStateFile;
	string fileNameL, prefixL = "LastState_", postfix = ".bin";
	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
	{
		fileNameL = prefixL + exp + toString(block_numb) + postfix;
		LastStateFile = fopen(fileNameL.c_str(), "rb");
		for (int64_t i = 0; i < N; i++)
		{
			fscanf(LastStateFile, "%lf %lf", &X0[i * 2 + block_numb * 2 * N], &X0[i * 2 + 1 + block_numb * 2 * N]);
		}
		fclose(LastStateFile);
	}
}

void InitPotential_BACKWARD(double *EPS, string exp)
{
	FILE *EPSFile;
	string fileNameE, prefixE = "EPS_", postfix = ".bin";
	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
	{
		fileNameE = prefixE + exp + toString(block_numb) + postfix;
		EPSFile = fopen(fileNameE.c_str(), "rb");
		for (int64_t i = 0; i < N; i++)
		{
			fscanf(EPSFile, "%lf", &EPS[i + block_numb*N]);
		}
		fclose(EPSFile);
	}
}
// END: Initialization

// main cycle
int main(int argc, char **argv)
{
	srand((unsigned int)time(NULL));

	clock_t startTime, fullTime;
	startTime = clock();

	double *X0, *EPS, *Vars, *Measures;

	EPS	 = new double[N*BLOCKS];
	X0	 = new double[N * 2 * BLOCKS];
	Vars = new double[12];

	// initialize experiment
	double	E0		= 0.2, 
			min		= 0.5, 
			max		= 1.5,
			c1		= (1 - pow(3.0, -0.5)) / 2.0,
			c2		= pow(3.0, -0.5),
			d1		= 0.5,
			g		= (2 - pow(3.0, 0.5)) / 24.0,
			t_start = 0,
			t_end	= 1,
			dt		= 1,
			n_step	= 20,
			t_step	= dt / n_step,
			start_count_max = 1.5;

	// Generate t_end without damn sequence of zeros
	for (int i = 0; i < 7; i++, t_end *= 10) {}

	int64_t M_size = -1;		

#ifdef MEASURES		
	double counter = 0;
	double counterMax = start_count_max;
	int64_t ALL_steps = round((t_end - t_start) / t_step);
	double t1 = t_start, t2 = t_start;
	bool flag = 0;
	for (int64_t iter = 0; iter < ALL_steps; iter++)
	{
		if (t2 < t_start + dt - t_step)
		{
			t2 += t_step;
		}
		else
		{
			t2 = t_start;
			t1 += dt;
		}
		if (counter > counterMax)
		{
			M_size++;
			counter = 0;
			counterMax *= start_count_max;
			flag = 1;
		}
		else counter++;
	}
	M_size += 2;
	Measures = new double[8 * M_size * BLOCKS];

#endif

#ifdef BKWD
	// inverse time step 
	dt = -1;
	t_step = dt / n_step;
#endif
	// calculate coefficients
	c1 = c1 * t_step;
	d1 = d1 * t_step;
	c2 = c2 * t_step;
	g = -1 * t_step * t_step * t_step * g / 2;

	// keep parameters for gpu
	Vars[0] = t_start;
	Vars[1] = t_end;
	Vars[2] = dt;
	Vars[3] = t_step;
	Vars[4] = c1;
	Vars[5] = c2;
	Vars[6] = d1;
	Vars[7] = g;
	Vars[8] = E0;
	Vars[9] = M_size;
#ifdef FWRD
	Vars[10] = t_end;//real_end
#else
	Vars[10] = t_start;//real_end
#endif		
	Vars[11] = start_count_max;

	// This will pick the best possible CUDA capable device
	int deviceNUM;
	cudaGetDeviceCount(&deviceNUM);
	if (deviceNUM < 1)
	{
		printf("no CUDA capable devices were detected\n");
		return 1;
	}

	//Get GPU information
	printf("number of CUDA devices:\t%d\n", deviceNUM);
	for (int devID = 0; devID < deviceNUM; devID++)
	{
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop, devID);
		int64_t global_memory = static_cast<int64_t>(dprop.totalGlobalMem);
		printf("   %d: n=%s g=%" PRId64 " sh=%i\n", devID, dprop.name, global_memory, dprop.sharedMemPerBlock);
	}

	// initialize energy values
	int64_t en_count = -1;
	double  energs[] = { 0.2, 0.3, 0.4, 0.08, 0.05, 0.03, 0.02, 0.008, 0.002 }, energ = 0.0;

	string fileNameM, fileNameT, fileNameE, fileNameL;
	fileNameM.reserve(20);
	fileNameT.reserve(20);
	fileNameL.reserve(20);
	fileNameE.reserve(20);

	// for return status
	cudaError_t cudaStatus;

#ifdef FWRD

	string prefixT = "Time_", prefixM = "Measures_", prefixE = "EPS_", prefixL = "LastState_", postfix = ".bin";
	clock_t time1, time2;
	for (en_count = START_ENERGY_NUMBER, energ = energs[en_count]; en_count<END_ENERGY_NUMBER; en_count++, energ = energs[en_count]) // iterate by enegry number
	{
		for (int64_t j = 0; j<EXPERIMENTS_PER_ENERGY_VALUE; j++) // iterate by experiment number
		{
			time1 = clock();
			E0 = energ;
			InitPoint(X0, E0);
			Vars[8] = E0;
			InitPotential(EPS, min, max);
			Vars[0] = t_start;
			Vars[1] = t_end;
			FILE *EPSFile;
			for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
			{

				fileNameE = prefixE + toString(en_count) + "_" + toString(j) + "_" + toString(block_numb) + postfix;
				EPSFile = fopen(fileNameE.c_str(), "wb");
				for (int64_t i = 0; i < N; i++)
				{
					fprintf(EPSFile, "%f\n", EPS[i + block_numb*N]);
				}
				fclose(EPSFile);
			}
			fileNameM = prefixM + toString(en_count) + "_" + toString(j);
			fileNameT = prefixT + toString(en_count) + "_" + toString(j) + postfix;
			fileNameL = prefixL + toString(en_count) + "_" + toString(j);

			cudaStatus = TakeMeAnswer(X0, EPS, Measures, M_size, Vars, (char *)fileNameM.c_str(), (char *)fileNameT.c_str(), (char *)fileNameL.c_str());

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Integration process failed!\n");
				return 1;
			}

			time2 = clock();
			printf("en_count=%i\texp_numb=%i\ttime=%f\n", en_count, j, (time2 - time1) / CLOCKS_PER_SEC);
		}
	}

#elif BKWD
	string prefixT = "Time_", prefixM = "BKWD_Measures_", prefixE = "EPS_", prefixL = "BKWD_LastState_", postfix = ".bin";


	for (en_count = START_ENERGY_NUMBER, energ = energs[en_count]; en_count<END_ENERGY_NUMBER; en_count++, energ = energs[en_count]) // iterate by enegry number
	{
		for (int64_t j = 0; j<EXPERIMENTS_PER_ENERGY_VALUE; j++) // iterate by experiment number
		{
			E0 = energ;
			string current_exp = toString(en_count) + "_" + toString(j) + "_";
			InitPoint_BACKWARD(X0, current_exp);
			Vars[8] = E0;
			InitPotential_BACKWARD(EPS, current_exp);
			Vars[0] = t_start;
			Vars[1] = t_end;

			fileNameM = prefixM + toString(en_count) + "_" + toString(j);
			fileNameT = prefixT + toString(en_count) + "_" + toString(j) + postfix;
			fileNameL = prefixL + toString(en_count) + "_" + toString(j);

			cudaStatus = TakeMeAnswer(X0, EPS, Measures, M_size, Vars, (char *)fileNameM.c_str(), (char *)fileNameT.c_str(), (char *)fileNameL.c_str());

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Integration process failed!\n");
				return 1;
			}

			printf("en_count=%i\texp_numb=%i\n", en_count, j);
		}
		printf("done: %i\n", en_count);
	}
#endif


	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Integration process failed!\n");
		return 1;
	}

	// Dump timing to file
	FILE *TimeFile;
	fullTime = clock();
	int64_t full_calculating_time = fullTime - startTime;
	TimeFile = fopen("Time.bin", "wb");
	fprintf(TimeFile, "%f", full_calculating_time / CLOCKS_PER_SEC);
	fclose(TimeFile);

	// Free CPU resources
	delete EPS;
	delete X0;
	delete Vars;

	// Free GPU resources
	cudaFree(dev_X0);
	cudaFree(dev_TempPoint);
	cudaFree(dev_EPS);
	cudaFree(dev_Energy);
	cudaFree(dev_Vars);
	cudaFree(dev_Measures);
	cudaFree(dev_Counters);

	return 0;
}

// Collecting answer for every experiment
cudaError_t TakeMeAnswer(double *X0, double *EPS, double *Measures, int64_t M_size, double *Vars, char* fileNameM, char* fileNameT, char* fileNameL)
{
	double *Counters = new double[4];
	double *Energy = new double[N];
	double *TempPoint = new double[2 * N];
	Counters[0] = 0;
	Counters[1] = Vars[11];
	Counters[2] = -1;
	Counters[3] = 0;


	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate memory on GPU once

	if (!dev_X0)
	{
		cudaStatus = cudaMalloc((void**)&dev_X0, 2 * N * BLOCKS * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_X0 failed!");
			return cudaStatus;
		}
	}

	if (!dev_TempPoint)
	{
		cudaStatus = cudaMalloc((void**)&dev_TempPoint, 2 * N * BLOCKS * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_TempPoint failed!");
			return cudaStatus;
		}
	}

	if (!dev_EPS)
	{
		cudaStatus = cudaMalloc((void**)&dev_EPS, N * BLOCKS * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_EPS failed!");
			return cudaStatus;
		}
	}

	if (!dev_Energy)
	{
		cudaStatus = cudaMalloc((void**)&dev_Energy, N * BLOCKS * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_Energy failed!");
			return cudaStatus;
		}
	}

	if (!dev_Vars)
	{
		cudaStatus = cudaMalloc((void**)&dev_Vars, 12 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_Vars failed!");
			return cudaStatus;
		}
	}

#ifdef MEASURES
	if (!dev_Measures)
	{
		cudaStatus = cudaMalloc((void**)&dev_Measures, 8 * M_size * BLOCKS * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_Measures failed!");
			return cudaStatus;
		}

	}
#endif

	if (!dev_Counters)
	{
		cudaStatus = cudaMalloc((void**)&dev_Counters, 4 * sizeof(double));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc dev_Counters failed!");
			return cudaStatus;
		}
	}

	// Copy memory from CPU to GPU

	cudaStatus = cudaMemcpy(dev_X0, X0, 2 * N * BLOCKS * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_X0, X0 failed!");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_EPS, EPS, N * BLOCKS * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_EPS, EPS failed!");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_Counters, Counters, 4 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_Counters, Counters failed!");
		return cudaStatus;
	}


	double t_start = Vars[0], t_end = Vars[1];

	// swap start and end if it's backward run
#ifdef BKWD
	t_start = Vars[1];
	t_end = Vars[0];
	Vars[10] = Vars[0];

#endif

	// Need to decrease kernel execution time, so I split it to steps
	// not the best way, but...
#ifdef FWRD
	int64_t colSteps = (int64_t)(t_end / STEP), currStep = 0;
	for (double time = t_start; time < t_end; time += STEP)
	{
		Vars[0] = time;
		if (time + STEP < t_end)
			Vars[1] = time + STEP;
		else
			Vars[1] = t_end;
#elif defined BKWD
	int64_t colSteps = (int64_t)(t_start / STEP), currStep = 0;

	for (double time = t_start; time > t_end; time -= STEP)
	{
		Vars[0] = time;
		if (time - STEP > t_end)
			Vars[1] = time - STEP;
		else
			Vars[1] = t_end;
#endif

		currStep++;
		cudaStatus = cudaMemcpy(dev_Vars, Vars, 12 * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy dev_Vars, Vars failed!");
			return cudaStatus;
		}

		dim3 grid, block;
		grid.x = BLOCKS;
		block.x = N;
		size_t sharedMemoryNumb = 7 * N * sizeof(double);

		SABAKernel << < grid, block, sharedMemoryNumb >> > (dev_X0, dev_TempPoint, dev_EPS, dev_Energy, dev_Measures, dev_Vars, dev_Counters);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "SABAKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SABAKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}


#ifdef MEASURES
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Measures, dev_Measures, 8 * M_size * BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Measures, dev_Measures failed!");
		return cudaStatus;
	}
#endif
	
	cudaStatus = cudaMemcpy(X0, dev_X0, 2 * N * BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy dev_X0 to X0 failed!");
		return cudaStatus;
	}

	string tmp;
	string LASTSTATEFILE(fileNameL);
	string MEASURESFILE(fileNameM);
	tmp.reserve(20);
	LASTSTATEFILE.reserve(20);
	MEASURESFILE.reserve(20);
	LASTSTATEFILE = string(fileNameL);
	MEASURESFILE = string(fileNameM);

	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
	{
		FILE *LastStateFile = nullptr;
		tmp = LASTSTATEFILE;
		tmp += "_";
		tmp += toString(block_numb);
		tmp += ".bin";
		LastStateFile = fopen(tmp.c_str(), "wb");
		for (int64_t i = 0; i < 2 * N; i++)
		{
			fprintf(LastStateFile, "%.16f ", X0[i + 2 * N * block_numb]);
			if (!!(i & 1))
				fprintf(LastStateFile, "\n");
		}
		fflush(LastStateFile);
		fclose(LastStateFile);
	}

#ifdef MEASURES
	for (int64_t block_numb = 0; block_numb < BLOCKS; block_numb++)
	{
		FILE *MeasuresFile;
		tmp = MEASURESFILE;
		tmp += "_";
		tmp += toString(block_numb);
		tmp += ".bin";
		MeasuresFile = fopen(tmp.c_str(), "wb");
		for (int64_t i = 0; i < 8 * M_size; i++)
		{
			fprintf(MeasuresFile, "%.16f ", Measures[i + 8 * M_size * block_numb]);
			if ((i & 7) == 7)
				fprintf(MeasuresFile, "\n");
		}
		fflush(MeasuresFile);
		fclose(MeasuresFile);
	}
#endif

	// free recources
	delete Counters;
	delete Energy;
	delete TempPoint;
	return cudaStatus;
}
