#pragma once

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <benchmark/benchmark.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace kumo {

class ScanCL {
public:
  ScanCL()
      : platform_(nullptr), context_(nullptr), device_(nullptr),
        queue_(nullptr), kernel_(nullptr), program_(nullptr) {};
  ~ScanCL() { UnInit(); };

  bool Init();
  void UnInit();

  bool BuildKernel(const std::string &source_path, const char *kernel_func_name,
                   cl_kernel *out_kernel, cl_program *out_program);

  bool Run(const std::vector<int> &input, const std::vector<int> &output,
           size_t batch_size);

private:
  bool ExclusiveScan(cl_command_queue queue, cl_mem input_buffer,
                     cl_mem temp_buffer, cl_uint batch_size,
                     cl_uint array_length);

private:
  cl_platform_id platform_;
  cl_context context_;
  cl_device_id device_;
  cl_command_queue queue_;
  cl_program program_;
  cl_kernel kernel_;
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
      "gaussian_blur_cols", &kernel_, &program_);
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
  if (program_)
    clReleaseProgram(program_);
  if (queue_)
    clReleaseCommandQueue(queue_);
  if (context_)
    clReleaseContext(context_);
  // device_ 和 platform_ 不需要释放
  kernel_ = nullptr;
  program_ = nullptr;
  queue_ = nullptr;
  context_ = nullptr;
  device_ = nullptr;
  platform_ = nullptr;
}

inline bool ScanCL::ExclusiveScan(cl_command_queue queue, cl_mem input_buffer,
                                  cl_mem output_buffer, cl_uint batch_size,
                                  cl_uint array_length) {
  cl_int err;

  int arg_index = 0;
  err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem),
                        (void *)&input_buffer);
  err = clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem),
                       (void *)&output_buffer);
  err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_mem), (void *)&kernel_);
  err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_uint),
                        (void *)&batch_size);
  err |= clSetKernelArg(kernel_, arg_index++, sizeof(cl_uint),
                        (void *)&array_length);
  if (err != CL_SUCCESS) {
    std::cerr << "RunKernel failed" << std::endl;
    return false;
  }

  size_t globalWorkSize[2] = {(size_t)array_length};
  size_t localWorkSize[2] = {(size_t)batch_size};
  err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, globalWorkSize,
                               localWorkSize, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "clEnqueueNDRangeKernel failed return " << err << std::endl;
  }

  return true;
}

inline bool ScanCL::Run(const std::vector<int> &input,
                        const std::vector<int> &output, size_t batch_size) {

  cl_int err = CL_SUCCESS;
  cl_mem input_buf =
      clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     input.size() * sizeof(int), (void *)input.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer input_buf failed return " << err << std::endl;
    return false;
  }

  cl_mem output_buf =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, output.size() * sizeof(int),
                     nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer output_buf failed return " << err << std::endl;
    return false;
  }

  ExclusiveScan(queue_, input_buf, output_buf, input.size(), batch_size);

  clFinish(queue_);

  std::vector<int> output_host(input.size());
  err = clEnqueueReadBuffer(queue_, output_buf, CL_TRUE, 0,
                            output_host.size() * sizeof(int),
                            output_host.data(), 0, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    std::cerr << "clEnqueueReadBuffer failed return " << err << std::endl;
  }

  clReleaseMemObject(input_buf);
  clReleaseMemObject(output_buf);
  return true;
}

} // namespace kumo