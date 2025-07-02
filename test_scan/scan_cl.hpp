#pragma once

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace kumo {

class ScanCL {
public:
  ScanCL()
      : platform_(nullptr), context_(nullptr), device_(nullptr),
        queue_(nullptr), kernel_(nullptr), kernel_uniform_add_(nullptr), program_(nullptr) {};
  ~ScanCL() { UnInit(); };

  bool Init();
  void UnInit();

  bool Run(const std::vector<int> &input, std::vector<int> &output, const int tile_size);

private:
  bool BuildKernel(const std::string &source_path, const char *kernel_func_name,
                   cl_kernel *out_kernel, cl_program *out_program);

  bool ExclusiveScan(cl_command_queue queue, cl_mem data, cl_mem tile_sum, int N, int TILE_SIZE);

private:
  cl_platform_id platform_;
  cl_context context_;
  cl_device_id device_;
  cl_command_queue queue_;
  cl_program program_;
  cl_kernel kernel_;
  cl_kernel kernel_uniform_add_;
};

inline bool ScanCL::Init() {
  cl_int err;

  // Discover avaliable OpenCL platform
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(1, &platform_, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    std::cerr << "clGetPlatformIDs error return " << err << std::endl;
    return false;
  }

  cl_uint num_devices = 0;
  err =
      clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) {
    std::cerr << "clGetDeviceIDs error return " << err << std::endl;
    return false;
  }

  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateContext error return " << err << std::endl;
    return false;
  }

#if CL_TARGET_OPENCL_VERSION >= 200
  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
                                 0};
  queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &err);
#else
  queue_ =
      clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
  if (err != CL_SUCCESS) {
#if CL_TARGET_OPENCL_VERSION >= 200
    std::cerr << "clCreateCommandQueueWithProperties error return " << err
              << std::endl;
#else
    std::cerr << "clCreateCommandQueue error return " << err << std::endl;
#endif
    return false;
  }

  BuildKernel(
      "/home/kumo/dev/hello_ocl_runtime/test_scan/scan.cl",
      "scan", &kernel_, &program_);

  BuildKernel(
      "/home/kumo/dev/hello_ocl_runtime/test_scan/scan.cl",
      "uniform_add", &kernel_uniform_add_, &program_);
  return true;
}

inline bool ScanCL::BuildKernel(const std::string &source_path,
                                const char *kernel_func_name,
                                cl_kernel *out_kernel,
                                cl_program *out_program) {
  // read .cl file
  std::ifstream file(source_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open OpenCL source file: " << source_path
              << std::endl;
    return false;
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  std::string source_code = oss.str();
  const char *source_ptr = source_code.c_str();

  // create program
  cl_int err = 0;
  cl_program program =
      clCreateProgramWithSource(context_, 1, &source_ptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateProgramWithSource error return " << err << std::endl;

    clReleaseProgram(program);
    program_ = nullptr;
    return false;
  }

  err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);

    std::vector<char> build_log(log_size);
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size,
                          build_log.data(), nullptr);

    std::cerr << "Build failed with error code " << err << ":\n"
              << build_log.data() << std::endl;
    return false;
  }

  cl_kernel kernel = clCreateKernel(program, kernel_func_name, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateKernel " << kernel_func_name << " error return "
              << err << std::endl;

    clReleaseProgram(program);
    program_ = nullptr;
    return false;
  }

  *out_kernel = kernel;
  if (out_program)
    *out_program = program;
  else
    clReleaseProgram(program);

  // std::cout << "Kernel '" << kernel_func_name << "' built successfully from"
  // << source_path << std::endl;

  return true;
}

inline void ScanCL::UnInit() {
  if (kernel_)
    clReleaseKernel(kernel_);
  if (kernel_uniform_add_)
    clReleaseKernel(kernel_uniform_add_);
  if (program_)
    clReleaseProgram(program_);
  if (queue_)
    clReleaseCommandQueue(queue_);
  if (context_)
    clReleaseContext(context_);
  // device_ 和 platform_ 不需要释放
  kernel_ = nullptr;
  kernel_uniform_add_ = nullptr;
  program_ = nullptr;
  queue_ = nullptr;
  context_ = nullptr;
  device_ = nullptr;
  platform_ = nullptr;
}

inline bool ScanCL::ExclusiveScan(cl_command_queue queue, cl_mem data, cl_mem tile_sum, int N, int TILE_SIZE) {
  cl_int err;
  std::vector<int> test_1(N, 0);
  int tile_sum_size = (N + TILE_SIZE - 1) / TILE_SIZE;
  std::vector<int> test_2(tile_sum_size, 0);
  std::vector<int> test_3(tile_sum_size, 0);
  std::vector<int> test_4(N, 0);
  {
    int arg_index = 0;
    err  = clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void *)&data);
    err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void *)&tile_sum);
    err |= clSetKernelArg(kernel_, arg_index++, TILE_SIZE * sizeof(int), nullptr);
    err |= clSetKernelArg(kernel_, arg_index++, sizeof(int), (void *)&N);
    if (err != CL_SUCCESS) {
      std::cerr << "RunKernel failed" << std::endl;
      return false;
    }

    size_t globalWorkSize = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t localWorkSize = TILE_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &globalWorkSize,
                                &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue_);
    if (err != CL_SUCCESS) {
      std::cerr << "clEnqueueNDRangeKernel failed return " << err << std::endl;
    }


    err = clEnqueueReadBuffer(queue_, data, CL_TRUE, 0,
                              test_1.size() * sizeof(int), test_1.data(), 0,
                              nullptr, nullptr);
    err = clEnqueueReadBuffer(queue_, tile_sum, CL_TRUE, 0,
                              test_2.size() * sizeof(int), test_2.data(), 0,
                              nullptr, nullptr);
    std::cout << "finish tile scan" << std::endl;
  }

  {
    int arg_index = 0;
    int tile_sum_size = (N + TILE_SIZE - 1) / TILE_SIZE;
    err  = clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void*)&tile_sum);
    err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void*)NULL);
    err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void*)NULL);
    err |= clSetKernelArg(kernel_, arg_index++, sizeof(int), (void*)&tile_sum_size);
    if (err != CL_SUCCESS) {
      std::cerr << "RunKernel failed" << std::endl;
      return false;
    }
    
    size_t localWorkSize = tile_sum_size;
    size_t globalWorkSize = localWorkSize;
    if (globalWorkSize > TILE_SIZE) {
      std::cerr << "invalid size! globalWorkSize=" << globalWorkSize << " TILE_SIZE=" << TILE_SIZE << std::endl;
      return false;
    }
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &globalWorkSize,
                                 &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue_);
    if (err != CL_SUCCESS) {
      std::cerr << "clEnqueueNDRangeKernel failed return " << err << std::endl;
    }
    err = clEnqueueReadBuffer(queue_, tile_sum, CL_TRUE, 0,
      test_3.size() * sizeof(int),
      test_3.data(), 0, nullptr, nullptr);
    std::cout << "finish offset scan" << std::endl;
  }

  {
    int arg_index = 0;
    err  = clSetKernelArg(kernel_uniform_add_, arg_index++, sizeof(cl_mem), (void*)&data);
    err |= clSetKernelArg(kernel_uniform_add_, arg_index++, sizeof(cl_mem), (void*)&tile_sum); 
    err |= clSetKernelArg(kernel_uniform_add_, arg_index++, sizeof(int), (void*)&N);
    err |= clSetKernelArg(kernel_uniform_add_, arg_index++, sizeof(int), (void*)&TILE_SIZE);
    size_t globalWorkSize = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    size_t localWorkSize = TILE_SIZE;
    clEnqueueNDRangeKernel(queue, kernel_uniform_add_, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue_);
    if (err != CL_SUCCESS) {
      std::cerr << "clEnqueueNDRangeKernel failed return " << err << std::endl;
    }
    err = clEnqueueReadBuffer(queue_, data, CL_TRUE, 0,
      test_4.size() * sizeof(int),
      test_4.data(), 0, nullptr, nullptr);
    std::cout << "finish uniform add" << std::endl;
  }


  return true;
}

inline bool ScanCL::Run(const std::vector<int> &input,
                        std::vector<int> &output,
                        const int tile_size) {

  cl_int err = CL_SUCCESS;
  
  output = input;

  cl_mem input_buf =
      clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     output.size() * sizeof(int), (void *)output.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer data failed return " << err << std::endl;
    return false;
  }

  std::vector<int> temp_vector(tile_size, 0);
  cl_mem temp_buffer = 
    clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tile_size * sizeof(int), temp_vector.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer temp_buffer failed return " << err << std::endl;
    return false;
  }

  ExclusiveScan(queue_, input_buf, temp_buffer, input.size(), tile_size);


  err = clEnqueueReadBuffer(queue_, input_buf, CL_TRUE, 0,
                            output.size() * sizeof(int),
                            output.data(), 0, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseMemObject(temp_buffer);
    std::cerr << "clEnqueueReadBuffer failed return " << err << std::endl;
  }

  clReleaseMemObject(input_buf);
  clReleaseMemObject(temp_buffer);
  return true;
}

} // namespace kumo