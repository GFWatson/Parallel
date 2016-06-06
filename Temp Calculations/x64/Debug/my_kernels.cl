

//sum vector
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//loop through local data and sum
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			scratch[lid] += scratch[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//add sums of all workgroups
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

//find min
__kernel void find_min(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int curMin = 1000;

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//loop through local data and find min
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if(curMin > scratch[lid]) {
				curMin = scratch[lid];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//after all are complete
	//find min of all workgroups
	barrier(CLK_GLOBAL_MEM_FENCE);
	atomic_min(&B[0], curMin);
}

//find max
__kernel void find_max(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int curMax = 1000;

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//loop through local data and find max
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			if(curMax < scratch[lid]) {
				curMax = scratch[lid];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//after all are complete
	//find max of all workgroups
	barrier(CLK_GLOBAL_MEM_FENCE);
	atomic_max(&B[0], curMax);
}

//histogram
__kernel void hist_simple(__global const int* A, __global int* H, __local int* scratch, int minTemp, int maxTemp, int numOfBins) { 
	int id = get_global_id(0);
	//assumes that H has been initialised to 0
	int bin_index;//take value as a bin index
	//find distance between categories
	//assumes even distance
	int stepNum = (maxTemp - minTemp) / numOfBins;
	//scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);
	
	//categories
	//depends of num of bins, found using steps
	if(A[id] <= (minTemp + stepNum)){
		bin_index = 0;
	}
	else if(A[id] > (minTemp + stepNum) && A[id] < (minTemp + (stepNum*2)) && numOfBins > 2) {
		bin_index = 1;
	}
	else if(A[id] > (minTemp + (stepNum*2)) && A[id] < (minTemp + (stepNum*3)) && numOfBins > 3) {
		bin_index = 2;
	}
	else if(A[id] > (minTemp + (stepNum*3)) && A[id] < (minTemp + (stepNum*4)) && numOfBins > 4) {
		bin_index = 3;
	}
	else {
		bin_index = 4;
	}

	//increment accordingly
	atomic_inc(&H[bin_index]);
}

//standard deviation
__kernel void find_sd(__global const int* A, __global int* B, __local int* scratch, int avg) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int num = 0;
	int sum = 0;
	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	//loop through local data
	//sum((individual - average)^2)
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) {
			num = scratch[lid] -= avg;
			num *= num;
			sum += num;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//sum of individual workgroups
	if (!lid) {
		atomic_add(&B[0],sum);
	}
}