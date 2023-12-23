////////////////////////////////////////////////////////////////////
//File: flip_environ.c
//
//Description: base file for environment exercises with openCL
//
// 
////////////////////////////////////////////////////////////////////
#define cimg_use_jpeg
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <thread>
#include "CImg.h"
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif


using namespace cimg_library;

enum {
  HALF_APPROACH,
  CHUNK_APPROACH
};

typedef struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
} uchar3;

struct full_device_context {
  cl_context context; 
  cl_command_queue command_queue; 
  cl_program program; 
  cl_kernel kernel; 
  size_t global_size; 
  size_t local_size; 
};

  
// check error, in such a case, it exits
// g++ basic_environ.c -o basic -lOpenCL

void cl_error(cl_int code, const char *string){
	if (code != CL_SUCCESS){
		printf("%d - %s\n", code, string);
	    exit(-1);
	}
}
////////////////////////////////////////////////////////////////////////////////

void initArray(unsigned char *array, CImg<unsigned char> img) {
    for (int y = 0; y < img.height(); y++) {
        for (int x = 0; x < img.width(); x++) {
              array[(y * img.width() + x) * 3] = img(x, y, 0, 0);
              array[(y * img.width() + x) * 3 + 1] = img(x, y, 0, 1);
              array[(y * img.width() + x) * 3 + 2] = img(x, y, 0, 2);
        }
    }
}

void convertImage(unsigned char *array, CImg<unsigned char> &img) {
    for (int y = 0; y < img.height(); y++) {
        for (int x = 0; x < img.width(); x++) {
               img(x, y, 0, 0) = array[(y * img.width() + x) * 3];
               img(x, y, 0, 1) = array[(y * img.width() + x) * 3 + 1];
               img(x, y, 0, 2) = array[(y * img.width() + x) * 3 + 2];
        }
    }
}

// ################################  WORKLOAD BALANCE APPROACHES (FUNCTIONS) ################################ 

void half(int N_images, int& N_images_cpu, int& N_images_gpu) {
  N_images_cpu = N_images / 2;
  N_images_gpu = N_images - N_images_cpu;
}

void divideInChunks(int N_images, int& N_images_cpu, int& N_images_gpu, float time_one_chunk_cpu, float time_one_chunk_gpu) {
  int cpu_chunk_size = 1;

  int gpu_chunk_size = time_one_chunk_cpu / time_one_chunk_gpu;
}

float timeOneChunk(full_device_context device_context, CImg<unsigned char> img) {
  // ################################  CREATE INPUT AND OUTPUT ARRAYS HOST MEMORY  ################################ 
  unsigned char image_data[img.size()];
  initArray(image_data, img);
  if (image_data == NULL) {
      perror("Failed to allocate memory for image_data");
      exit(EXIT_FAILURE);
  }
  int err;
  cl_mem img_buffer;
  img_buffer = clCreateBuffer(device_context.context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img.size(), NULL, &err);
  cl_error(err, "Failed to create memory buffer at device\n");

    // ################################ COPY DATA FROM HOST TO DEV  ################################
  // Measurement of bandwidth from mem to kernel bandwidth = bytes passed / time spent passing them ==> B/s
  
  // Write data into the memory object
  err = clEnqueueWriteBuffer(device_context.command_queue, img_buffer, CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a write command\n");
    

  // ################################ PASS ARGUMENTS  ################################
  unsigned int width = img.width();
  unsigned int height = img.height();

  // Set the arguments to the kernel
  err = clSetKernelArg(device_context.kernel, 0, sizeof(img_buffer), &img_buffer);
  cl_error(err, "Failed to set argument 0\n");
  err = clSetKernelArg(device_context.kernel, 1, sizeof(unsigned int), &width);
  cl_error(err, "Failed to set argument 1\n");
  err = clSetKernelArg(device_context.kernel, 2, sizeof(unsigned int), &height);
  cl_error(err, "Failed to set argument 2\n");
  
  // ################################ LAUNCH KERNEL FUNCTION ################################ 
  // Launch Kernel
  cl_event event;
  unsigned long start_kernel_time, end_kernel_time;
  err = clEnqueueNDRangeKernel(device_context.command_queue, device_context.kernel, 1, NULL, &device_context.global_size, NULL, 0, NULL, &event);
  cl_error(err, "Failed to launch kernel to the device\n");
  
  clWaitForEvents(1, &event);
  // Obtainint profiling times
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_kernel_time, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_kernel_time, NULL);

  unsigned long elapsed_time_kernel = end_kernel_time - start_kernel_time;
  float elapsed_time_kernel_seconds = (float)elapsed_time_kernel * 1e-9;

  // ################################ READ IMAGE (AUTOMATICALLY REPLACED) ################################ 
  //enqueue the order to read results form device memory
  err = clEnqueueReadBuffer(device_context.command_queue, img_buffer, CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a read command\n");
  
  // ################################ FREE MEM ################################ 
  clReleaseMemObject(img_buffer);
  return elapsed_time_kernel_seconds;
}

// ################################  PARALLEL KERNEL RUNNING ################################ 
void runKernel(int N_images, cl_context context, cl_command_queue command_queue, cl_program program, cl_kernel kernel, 
            size_t global_size, size_t local_size, int& err, CImg<unsigned char> img, bool isCpu) {

  // ################################  CREATE INPUT AND OUTPUT ARRAYS HOST MEMORY  ################################ 
  unsigned char image_data[img.size()];
  initArray(image_data, img);
  if (image_data == NULL) {
      perror("Failed to allocate memory for image_data");
      exit(EXIT_FAILURE);
  }

  cl_mem img_buffers[N_images];
  for (int i = 0; i < N_images; ++i) {
    img_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img.size(), NULL, &err);
    cl_error(err, "Failed to create memory buffer at device\n");

      // ################################ COPY DATA FROM HOST TO DEV  ################################
    // Measurement of bandwidth from mem to kernel bandwidth = bytes passed / time spent passing them ==> B/s
    
    // Write data into the memory object
    err = clEnqueueWriteBuffer(command_queue, img_buffers[i], CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data, 0, NULL, NULL);
    cl_error(err, "Failed to enqueue a write command\n");
    
  }

  // ################################ PASS ARGUMENTS  ################################
  unsigned int width = img.width();
  unsigned int height = img.height();
  for (int i = 0; i < N_images; ++i) {
    // Set the arguments to the kernel
    err = clSetKernelArg(kernel, 0, sizeof(img_buffers[i]), &img_buffers[i]);
    cl_error(err, "Failed to set argument 0\n");
    err = clSetKernelArg(kernel, 1, sizeof(unsigned int), &width);
    cl_error(err, "Failed to set argument 1\n");
    err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &height);
    cl_error(err, "Failed to set argument 2\n");
    
    // ################################ LAUNCH KERNEL FUNCTION ################################ 
    // Launch Kernel
    local_size = 128;
    global_size = (size_t)(img.size() / 3);
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    cl_error(err, "Failed to launch kernel to the device\n");
  }
  

  // ################################ READ IMAGE (AUTOMATICALLY REPLACED) ################################ 
  for (int i = 0; i < N_images; ++i) {
    //enqueue the order to read results form device memory
    err = clEnqueueReadBuffer(command_queue, img_buffers[i], CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data, 0, NULL, NULL);
    cl_error(err, "Failed to enqueue a read command\n");

    if (i == 0 || i == N_images - 1) {
      char file_name[50];  
      sprintf(file_name, "flipped%d.jpg", i);
      
      convertImage(image_data, img);
      img.save(file_name);
    }
  }
  
  
  // ################################ FREE MEM ################################ 
  for (int i = 0; i < N_images; ++i) clReleaseMemObject(img_buffers[i]);
  
  if (isCpu) std::cout << "Ended CPU" << std::endl;
  else std::cout << "Ended GPU" << std::endl;
}

int main(int argc, char** argv)
{
  int balance_approach = CHUNK_APPROACH; 
  if (argc < 6){ // 1 for adding gpu 0 for not adding it
    std::cout << "Usage: " << argv[0] << " <cpu_platform> <cpu_device> <gpu_platform> <gpu_device> <add_gpu>" << std::endl;
    return 1;
  }

  int cpu_platform = std::stoi(argv[1]);
  int cpu_device = std::stoi(argv[2]);
  int gpu_platform = std::stoi(argv[3]);
  int gpu_device = std::stoi(argv[4]);
  bool add_gpu = std::stoi(argv[5]) == 1 ? true : false;

  const unsigned int N = 100;
  int err;                            	// error code returned from api calls
  size_t t_buf = 50;			// size of str_buffer
  char str_buffer[t_buf];		// auxiliary buffer	
  size_t e_buf;				// effective size of str_buffer in use
	    
  size_t cpu_global_size;                      	// global domain size for our calculation
  size_t cpu_local_size;                       	// local domain size for our calculation

  size_t gpu_global_size;                      	// global domain size for our calculation
  size_t gpu_local_size;                       	// local domain size for our calculation

  const cl_uint num_platforms_ids = 10;				// max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];		// array of platforms
  cl_uint n_platforms;						// effective number of platforms in use
  const cl_uint num_devices_ids = 10;				// max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids];	// array of devices
  cl_uint n_devices[num_platforms_ids];				// effective number of devices in use for each platform
	
  cl_device_id cpu_device_id;             				// compute device id 
  cl_context cpu_context;                 				// compute context
  cl_command_queue cpu_command_queue;     				// compute command queue
  cl_program cpu_program;                         // compute program
  cl_kernel cpu_kernel;                           // compute kernel

  cl_device_id gpu_device_id;             				// compute device id 
  cl_context gpu_context;                 				// compute context
  cl_command_queue gpu_command_queue;     				// compute command queue
  cl_program gpu_program;                         // compute program
  cl_kernel gpu_kernel;                           // compute kernel

  // ################################ OVERALL TIME ################################ 
  clock_t start_time, end_time;
  float cpu_time;
  start_time = clock();

  // ################################ KERNEL TIME ################################ 
  cl_event events[1];
  unsigned long start_kernel_time, end_kernel_time;
    

  // 1. Scan the available platforms:
  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
  cl_error(err, "Error: Failed to Scan for Platforms IDs");
  printf("Number of available platforms: %d\n\n", n_platforms);

  for (int i = 0; i < n_platforms; i++ ){
    err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
    cl_error (err, "Error: Failed to get info of the platform\n");
    printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
  }
  printf("\n");
  // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...
	
  // 2. Scan for devices in each platform
  for (int i = 0; i < n_platforms; i++ ){
    err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_ALL, num_devices_ids, devices_ids[i], &(n_devices[i]));
    cl_error(err, "Error: Failed to Scan for Devices IDs");
    printf("\t[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);

    for(int j = 0; j < n_devices[i]; j++){
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_NAME, sizeof(str_buffer), str_buffer, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device name");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_NAME: %s\n", i, j,str_buffer);

      cl_uint max_compute_units_available;
      err = clGetDeviceInfo(devices_ids[i][j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units_available), &max_compute_units_available, NULL);
      cl_error(err, "clGetDeviceInfo: Getting device max compute units available");
      printf("\t\t [%d]-Platform [%d]-Device CL_DEVICE_MAX_COMPUTE_UNITS: %d\n\n", i, j, max_compute_units_available);
    }
  }	
  // ***Task***: print on the screen the cache size, global mem size, local memsize, max work group size, profiling timer resolution and ... of each device


  // 3. Create a context, one for the CPU and another one for the GPU
  cl_context_properties cpu_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[cpu_platform], 0};
  cpu_context = clCreateContext(cpu_properties, n_devices[cpu_platform], devices_ids[cpu_platform], NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");

  if (add_gpu){
    cl_context_properties gpu_properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[gpu_platform], 0};
    gpu_context = clCreateContext(gpu_properties, n_devices[gpu_platform], devices_ids[gpu_platform], NULL, NULL, &err);
    cl_error(err, "Failed to create a compute context\n");
  }
  


  // 4. Create a command queue for both devices
  cl_command_queue_properties cpu_proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  cpu_command_queue = clCreateCommandQueueWithProperties(cpu_context, devices_ids[cpu_platform][cpu_device], cpu_proprt, &err);
  cl_error(err, "Failed to create a command queue\n");

  if (add_gpu){
    // 4. Create a command queue
    cl_command_queue_properties gpu_proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    gpu_command_queue = clCreateCommandQueueWithProperties(gpu_context, devices_ids[gpu_platform][gpu_device], gpu_proprt, &err);
    cl_error(err, "Failed to create a command queue\n");
  }

  // ################################ GET IMAGE ################################ 
  CImg<unsigned char> img("image.jpg");
  CImg<unsigned char> img_1("image1.jpg");
  
  // ################################ LOAD KERNEL ################################ 
  // Calculate size of the file
  FILE *fileHandler = fopen("kernel_flip.cl", "r");
  if (fileHandler == NULL) {
    printf("Unable to open file\n");
    return 1;
  }
  fseek(fileHandler, 0, SEEK_END);
  size_t fileSize = ftell(fileHandler);
  rewind(fileHandler);

  // read kernel source into buffer
  char * sourceCode = (char*) malloc(fileSize + 1);
  sourceCode[fileSize] = '\0';
  fread(sourceCode, sizeof(char), fileSize, fileHandler);
  fclose(fileHandler);

  // create program from buffer 
  const char* constSourceCode = (const char*)sourceCode;
  cpu_program = clCreateProgramWithSource(cpu_context, 1, &constSourceCode, NULL, &err);
  cl_error(err, "Failed to create program with source\n");
  free(sourceCode);

  if (add_gpu){
    gpu_program = clCreateProgramWithSource(gpu_context, 1, &constSourceCode, NULL, &err);
    cl_error(err, "Failed to create program with source\n");
    free(sourceCode);
  }
  

  // ################################ BUILD KERNEL ################################ 
  // Build the executable and check errors
  err = clBuildProgram(cpu_program, 1, devices_ids[cpu_platform], NULL, NULL, NULL);
  if (err != CL_SUCCESS){
    size_t len;
    char buffer[2048];

    printf("Error: Some error at building process.\n");
    // first call to determine the size of the build log
    clGetProgramBuildInfo(cpu_program, devices_ids[cpu_platform][cpu_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    buffer[len] = '\0';
    // second call to retrieve the actual log data
    clGetProgramBuildInfo(cpu_program, devices_ids[cpu_platform][cpu_device], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    printf("%s\n", buffer);
    exit(-1);
  }

  if (add_gpu){
    // Build the executable and check errors
    err = clBuildProgram(gpu_program, 1, devices_ids[gpu_platform], NULL, NULL, NULL);
    if (err != CL_SUCCESS){
      size_t len;
      char buffer[2048];

      printf("Error: Some error at building process.\n");
      // first call to determine the size of the build log
      clGetProgramBuildInfo(gpu_program, devices_ids[gpu_platform][gpu_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
      buffer[len] = '\0';
      // second call to retrieve the actual log data
      clGetProgramBuildInfo(gpu_program, devices_ids[gpu_platform][gpu_device], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
      printf("%s\n", buffer);
      exit(-1);
    }
  }

  // ################################  CREATE KERNEL ################################ 
  // Create a compute kernel with the program we want to run
  cpu_kernel = clCreateKernel(cpu_program, "image_flip", &err);
  cl_error(err, "Failed to create kernel from the program\n");

  if (add_gpu){
    // Create a compute kernel with the program we want to run
    gpu_kernel = clCreateKernel(gpu_program, "image_flip", &err);
    cl_error(err, "Failed to create kernel from the program\n");
  }

  int N_images = 2;

  // ################################  WORKLOAD BALANCE APPROACHES  ################################ 
  int N_images_cpu = 1;
  int N_images_gpu = 1;
  full_device_context cpu_context_struct;
  full_device_context gpu_context_struct;
  float time_one_chunk_cpu = 0;
  float time_one_chunk_gpu = 0;
  switch (balance_approach)
  {
  case HALF_APPROACH:
    half(N_images, N_images_cpu, N_images_gpu);
    break;
  case CHUNK_APPROACH:
    cpu_context_struct.command_queue = cpu_command_queue;
    cpu_context_struct.context = cpu_context;
    cpu_context_struct.global_size = cpu_global_size;
    cpu_context_struct.local_size = cpu_local_size;
    cpu_context_struct.program = cpu_program;
    cpu_context_struct.kernel = cpu_kernel;
    time_one_chunk_cpu = timeOneChunk(cpu_context_struct, img);

    
    gpu_context_struct.command_queue = gpu_command_queue;
    gpu_context_struct.context = gpu_context;
    gpu_context_struct.global_size = gpu_global_size;
    gpu_context_struct.local_size = gpu_local_size;
    gpu_context_struct.program = gpu_program;
    gpu_context_struct.kernel = gpu_kernel;
    time_one_chunk_gpu = timeOneChunk(gpu_context_struct, img);
    divideInChunks(N_images, N_images_cpu, N_images_gpu, time_one_chunk_cpu, time_one_chunk_gpu);
    break;
  default:
    return 1;
    break;
  }
  
  
  
  // GO PARALLEL NOW
  std::vector<std::thread> thread_vector;
  thread_vector.push_back(std::thread(runKernel, N_images_cpu, cpu_context, cpu_command_queue, cpu_program, cpu_kernel, 
                          cpu_global_size, cpu_local_size, std::ref(err), img, true));
  if (add_gpu){
  thread_vector.push_back(std::thread(runKernel, N_images_gpu, gpu_context, gpu_command_queue, gpu_program, gpu_kernel, 
                          gpu_global_size, gpu_local_size, std::ref(err), img, true));
  }

  for(size_t i = 0; i < thread_vector.size(); ++i) {
      thread_vector[i].join();
  }

  clReleaseProgram(cpu_program);
  clReleaseKernel(cpu_kernel);
  clReleaseCommandQueue(cpu_command_queue);
  clReleaseContext(cpu_context);

  if (add_gpu){
    clReleaseProgram(gpu_program);
    clReleaseKernel(gpu_kernel);
    clReleaseCommandQueue(gpu_command_queue);
    clReleaseContext(gpu_context);
  }

  return 0;
}
//g++ flip_environ_two_devices.cc -o flip_environ_two_devices -I "CImg-2" -lm -lpthread -lX11 -ljpeg -lOpenCL -std=c++11
//./flip_environ_two_devices
