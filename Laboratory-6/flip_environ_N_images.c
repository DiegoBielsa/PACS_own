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
#include "CImg.h"
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif


using namespace cimg_library;

typedef struct {
    unsigned char x;
    unsigned char y;
    unsigned char z;
} uchar3;

  
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

int main(int argc, char** argv)
{
  const unsigned int N = 100;
  int err;                            	// error code returned from api calls
  size_t t_buf = 50;			// size of str_buffer
  char str_buffer[t_buf];		// auxiliary buffer	
  size_t e_buf;				// effective size of str_buffer in use
	    
  size_t global_size;                      	// global domain size for our calculation
  size_t local_size;                       	// local domain size for our calculation

  const cl_uint num_platforms_ids = 10;				// max of allocatable platforms
  cl_platform_id platforms_ids[num_platforms_ids];		// array of platforms
  cl_uint n_platforms;						// effective number of platforms in use
  const cl_uint num_devices_ids = 10;				// max of allocatable devices
  cl_device_id devices_ids[num_platforms_ids][num_devices_ids];	// array of devices
  cl_uint n_devices[num_platforms_ids];				// effective number of devices in use for each platform
	
  cl_device_id device_id;             				// compute device id 
  cl_context context;                 				// compute context
  cl_command_queue command_queue;     				// compute command queue
  cl_program program;                         // compute program
  cl_kernel kernel;                           // compute kernel

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


  // 3. Create a context, with a device
  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[2], 0};
  context = clCreateContext(properties, n_devices[0], devices_ids[0], NULL, NULL, &err);
  cl_error(err, "Failed to create a compute context\n");

  // 4. Create a command queue
  cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
  cl_error(err, "Failed to create a command queue\n");

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
  program = clCreateProgramWithSource(context, 1, &constSourceCode, NULL, &err);
  cl_error(err, "Failed to create program with source\n");
  free(sourceCode);
  

  // ################################ BUILD KERNEL ################################ 
  // Build the executable and check errors
  err = clBuildProgram(program, 1, devices_ids[0], NULL, NULL, NULL);
  if (err != CL_SUCCESS){
    size_t len;
    char buffer[2048];

    printf("Error: Some error at building process.\n");
    // first call to determine the size of the build log
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    buffer[len] = '\0';
    // second call to retrieve the actual log data
    clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    printf("%s\n", buffer);
    exit(-1);
  }

  // ################################  CREATE KERNEL ################################ 
  // Create a compute kernel with the program we want to run
  kernel = clCreateKernel(program, "image_flip", &err);
  cl_error(err, "Failed to create kernel from the program\n");

  // ################################  CREATE INPUT AND OUTPUT ARRAYS HOST MEMORY  ################################ 
  const unsigned int N_images = 2;
  unsigned char image_data[N_images][img.size()];
  initArray(image_data[0], img);
  initArray(image_data[1], img_1);
  if (image_data == NULL) {
      perror("Failed to allocate memory for image_data");
      exit(EXIT_FAILURE);
  }

  // ################################ CREATE INPUT AND OUTPUT ARRAYS DEVICE MEMORY  ################################ 
  
  cl_mem img_buffers[N_images];
  for (int i = 0; i < N_images; ++i) {
    img_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * img.size(), NULL, &err);
    cl_error(err, "Failed to create memory buffer at device\n");

      // ################################ COPY DATA FROM HOST TO DEV  ################################
    // Measurement of bandwidth from mem to kernel bandwidth = bytes passed / time spent passing them ==> B/s
    
    // Write data into the memory object
    err = clEnqueueWriteBuffer(command_queue, img_buffers[i], CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data[i], 0, NULL, NULL);
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
    err = clEnqueueReadBuffer(command_queue, img_buffers[i], CL_TRUE, 0, sizeof(unsigned char) * img.size(), image_data[i], 0, NULL, NULL);
    cl_error(err, "Failed to enqueue a read command\n");

    
    char filename[50];  // Ajusta el tamaño según tus necesidades
    sprintf(filename, "flipped%d.jpg", i);
    
    convertImage(image_data[i], img);
    img.save(filename);
  }
  
  
  // ################################ FREE MEM ################################ 
  for (int i = 0; i < N_images; ++i) clReleaseMemObject(img_buffers[i]);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
//g++ flip_environ_N_images.c -o flip_environ_N_images -I "CImg-2" -lm -lpthread -lX11 -ljpeg -lOpenCL
//./flip_environ_N_images
