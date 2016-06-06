#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}



int main(int argc, char **argv) {
	//command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//computing device
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		
		//Part 4 - memory allocation
		
		//vector to hold file temps
		//read in file and store temperatures in vector
		//multiply floats to reduce data loss in conversion to integer
		std::ifstream file("..\\temp_lincolnshire.txt");
		string word;
		std::vector<int> fileContents;
		int num = 0;
		
		while (file >> word) {
			num++;
			if (num == 6) {
				num = 0;
				fileContents.push_back((stof(word) * 100));
			}
		}

		//set workgroup size
		size_t local_size = 32;

		//find padding size
		size_t padding_size = fileContents.size() % local_size;

		//if the number of temperature is not a multiple of the workgroup size
		//insert additional elements as 0
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<float> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			fileContents.insert(fileContents.end(), A_ext.begin(), A_ext.end());
		}

		//find num of temperatures
		size_t numOfTemps = fileContents.size();
		//find size of temperature vector
		size_t tempSize = numOfTemps * sizeof(int);
		//find number of groups needed
		size_t numOfGroups = numOfTemps / local_size;

		//vectors for stats
		std::vector<int> statsAvg{ 0 };
		std::vector<int> statsMin{ 0 };
		std::vector<int> statsMax{ 0 };
		std::vector<int> statsSD{ 0 };
		size_t statSize = 1 * sizeof(int);
		std::vector<int> histogram{ 0,0,0,0,0 };
		size_t histSize = 5 * sizeof(int);

		//get user input for bins
		std::cout << "Number of bins (min 2 - max 5): " << std::endl;
		string input = "2";
		std::cin >> input;
		int numOfBins = stoi(input);
		//some imput handling for safety
		if (numOfBins < 2 || numOfBins > 5) {
			numOfBins = 2;
		}

		//create buffers
		cl::Buffer buffer_A1(context, CL_MEM_READ_ONLY, tempSize);
		cl::Buffer buffer_A2(context, CL_MEM_READ_ONLY, tempSize);
		cl::Buffer buffer_A3(context, CL_MEM_READ_ONLY, tempSize);
		cl::Buffer buffer_A4(context, CL_MEM_READ_ONLY, tempSize);
		cl::Buffer buffer_A5(context, CL_MEM_READ_ONLY, tempSize);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, statSize);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, statSize);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, statSize);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, histSize);
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, histSize);


		//copy vectors into local memory
		queue.enqueueWriteBuffer(buffer_A1, CL_TRUE, 0, tempSize, &fileContents[0]);
		queue.enqueueWriteBuffer(buffer_A2, CL_TRUE, 0, tempSize, &fileContents[0]);
		queue.enqueueWriteBuffer(buffer_A3, CL_TRUE, 0, tempSize, &fileContents[0]);
		queue.enqueueWriteBuffer(buffer_A4, CL_TRUE, 0, tempSize, &fileContents[0]);
		queue.enqueueWriteBuffer(buffer_A5, CL_TRUE, 0, tempSize, &fileContents[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, statSize);
		queue.enqueueFillBuffer(buffer_C, 0, 0, statSize);
		queue.enqueueFillBuffer(buffer_D, 0, 0, statSize);
		queue.enqueueFillBuffer(buffer_F, 0, 0, statSize);
		queue.enqueueWriteBuffer(buffer_E, CL_TRUE, 0, histSize, &histogram[0]);
		
		//set up kernels and run
		
		//sum to find average
		cl::Kernel kernel_add = cl::Kernel(program, "reduce_add_4");
		kernel_add.setArg(0, buffer_A1);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, cl::Local(local_size*sizeof(int)));//local memory size

		//find minimum value
		cl::Kernel kernel_min = cl::Kernel(program, "find_min");
		kernel_min.setArg(0, buffer_A1);
		kernel_min.setArg(1, buffer_C);
		kernel_min.setArg(2, cl::Local(local_size*sizeof(int)));//local memory size
		
		//find maximum value
		cl::Kernel kernel_max = cl::Kernel(program, "find_max");
		kernel_max.setArg(0, buffer_A3);
		kernel_max.setArg(1, buffer_D);
		kernel_max.setArg(2, cl::Local(local_size*sizeof(int)));//local memory size
		
		//find sum, min, max before finding standard deviation and histogram
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(numOfTemps), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(numOfTemps), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(numOfTemps), cl::NDRange(local_size));
		
		//read data from buffers
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, statSize, &statsAvg[0]);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, statSize, &statsMin[0]);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, statSize, &statsMax[0]);
		
		
		//standard deviation
		int avgInt = (statsAvg[0] / numOfTemps);
		cl::Kernel kernel_sd = cl::Kernel(program, "find_sd");
		kernel_sd.setArg(0, buffer_A5);
		kernel_sd.setArg(1, buffer_F);
		kernel_sd.setArg(2, cl::Local(local_size*sizeof(int)));//local memory size
		kernel_sd.setArg(3, avgInt);

		queue.enqueueNDRangeKernel(kernel_sd, cl::NullRange, cl::NDRange(numOfTemps), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, statSize, &statsSD[0]);

		//histogram
		cl::Kernel kernel_hist = cl::Kernel(program, "hist_simple");
		kernel_hist.setArg(0, buffer_A4);
		kernel_hist.setArg(1, buffer_E);
		kernel_hist.setArg(2, cl::Local(local_size*sizeof(int)));//local memory size
		kernel_hist.setArg(3, statsMin[0]);
		kernel_hist.setArg(4, statsMax[0]);
		kernel_hist.setArg(5, numOfBins);

		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(numOfTemps), cl::NDRange(local_size));
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, histSize, &histogram[0]);

		//print data for user
		
		//average
		float avg = (statsAvg[0] / numOfTemps);
		avg /= 100;
		std::cout << "Average: " << avg << std::endl;
		
		//min
		float minTemp = (statsMin[0]);
		minTemp /= 100;
		std::cout << "Minimum: " << minTemp << std::endl;

		//max
		float maxTemp = (statsMax[0]);
		maxTemp /= 100;
		std::cout << "Maximum: " << maxTemp << std::endl;

		//standard deviation
		//using sum((individual - average)^2)
		// sqrt(sum / n-1)
		//DO NOT DIVIDE BY 100
		float sd = (statsSD[0] / (numOfTemps - 1));
		sd = sqrt(sd);
		std::cout << "Standard Deviation: " << sd << std::endl;

		//histogram
		//based on number of bins
		int stepNum = (maxTemp - minTemp) / numOfBins;

		if (numOfBins == 2) {
			std::cout << "\nHISTOGRAM\n" << std::endl;
			std::cout << "<= " << (minTemp + stepNum) << ": " << histogram[0] << std::endl;
			std::cout << "> " << (minTemp + stepNum) << ": " << histogram[4] << std::endl;
		}
		else if (numOfBins == 3) {
			std::cout << "\nHISTOGRAM\n" << std::endl;
			std::cout << "<= " << (minTemp + stepNum) << ": " << histogram[0] << std::endl;
			std::cout << "> " << (minTemp + stepNum) << " & <= " << (minTemp + (stepNum * 2)) << ": " << histogram[1] << std::endl;
			std::cout << "> " << (minTemp + (stepNum*2)) << ": " << histogram[4] << std::endl;
		}
		else if (numOfBins == 4) {
			std::cout << "\nHISTOGRAM\n" << std::endl;
			std::cout << "<= " << (minTemp + stepNum) << ": " << histogram[0] << std::endl;
			std::cout << "> " << (minTemp + stepNum) << " & <= " << (minTemp + (stepNum * 2)) << ": " << histogram[1] << std::endl;
			std::cout << "> " << (minTemp + (stepNum * 2)) << " & <= " << (minTemp + (stepNum * 3)) << ": " << histogram[2] << std::endl;
			std::cout << "> " << (minTemp + (stepNum * 3)) << ": " << histogram[4] << std::endl;
		}
		else {
			std::cout << "\nHISTOGRAM\n" << std::endl;
			std::cout << "<= " << (minTemp + stepNum) << ": " << histogram[0] << std::endl;
			std::cout << "> " << (minTemp + stepNum) << " & <= " << (minTemp + (stepNum * 2)) << ": " << histogram[1] << std::endl;
			std::cout << "> " << (minTemp + (stepNum * 2)) << " & <= " << (minTemp + (stepNum * 3)) << ": " << histogram[2] << std::endl;
			std::cout << "> " << (minTemp + (stepNum * 3)) << " & <= " << (minTemp + (stepNum * 4)) << ": " << histogram[3] << std::endl;
			std::cout << "> " << (minTemp + (stepNum * 4)) << ": " << histogram[4] << std::endl;
		}

		
		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("pause");

	return 0;
}