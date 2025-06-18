#pragma once

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <benchmark/benchmark.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace kumo {

class OpenCLSeperableConv {
public:
  OpenCLSeperableConv()
      : platform_(nullptr), context_(nullptr), device_(nullptr),
        queue_(nullptr), kernel_cols_(nullptr), kernel_rows_(nullptr),
        program_(nullptr), valid_(false) {};
  ~OpenCLSeperableConv() {
    UnInit();
  };

  bool Init();
  void UnInit();

  bool BuildKernel(const std::string& source_path, const char* kernel_func_name, cl_kernel* out_kernel, cl_program* out_program);
  bool RunKernel(cl_kernel kernel, cl_command_queue queue,
    cl_mem input_buffer, cl_mem output_buffer,
    cl_mem gaussian_kernel, cl_uint width, cl_uint height,
    cl_uint pitch, cl_uint k_w, cl_uint k_h);

  bool Run(const cv::Mat& input, const std::vector<float>& kernel, cv::Mat& output);
  bool IsValid() const;

private:
  cl_platform_id platform_;
  cl_context context_;
  cl_device_id device_;
  cl_command_queue queue_;
  cl_program program_;
  cl_kernel kernel_rows_;
  cl_kernel kernel_cols_;
  cl_kernel kernel_;
  bool valid_;
};

inline bool OpenCLSeperableConv::Init() {
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
    std::cerr << "clCreateCommandQueueWithProperties error return " << err << std::endl;
#else
    std::cerr << "clCreateCommandQueue error return " << err << std::endl;
#endif
    return false;
  }

  valid_ = true;

  BuildKernel(
    "/home/kumo/dev/hello_ocl_runtime/kernels/gaussian_blur.cl",
    "gaussian_blur", &kernel_, &program_);
  return true;

  return true;
}

inline bool OpenCLSeperableConv::BuildKernel(const std::string& source_path, const char* kernel_func_name, cl_kernel* out_kernel, cl_program* out_program) {
  // read .cl file
  std::ifstream file(source_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open OpenCL source file: " << source_path << std::endl;
    return false;
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  std::string source_code = oss.str();
  const char* source_ptr = source_code.c_str();

  // create program
  cl_int err = 0;
  cl_program program = clCreateProgramWithSource(context_, 1, &source_ptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateProgramWithSource error return " << err << std::endl;

    clReleaseProgram(program);
    program_ = nullptr;
    return false;
  }

  err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

      std::vector<char> build_log(log_size);
      clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);

      std::cerr << "Build failed with error code " << err << ":\n" << build_log.data() << std::endl;
      return false;
  }

  cl_kernel kernel = clCreateKernel(program, kernel_func_name, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateKernel " << kernel_func_name << " error return " << err << std::endl;

    clReleaseProgram(program);
    program_ = nullptr;
    return false;
  }

  *out_kernel = kernel;
  if (out_program) *out_program = program;
  else clReleaseProgram(program);

  // std::cout << "Kernel '" << kernel_func_name << "' built successfully from" << source_path << std::endl;

  return true;
}

inline void OpenCLSeperableConv::UnInit() {
  if (kernel_cols_) clReleaseKernel(kernel_cols_);
  if (kernel_rows_) clReleaseKernel(kernel_rows_);
  if (kernel_) clReleaseKernel(kernel_);
  if (program_) clReleaseProgram(program_);
  if (queue_) clReleaseCommandQueue(queue_);
  if (context_) clReleaseContext(context_);
  // device_ 和 platform_ 不需要释放
  kernel_cols_ = nullptr;
  kernel_rows_ = nullptr;
  kernel_ = nullptr;
  program_ = nullptr;
  queue_ = nullptr;
  context_ = nullptr;
  device_ = nullptr;
  platform_ = nullptr;
  valid_ = false;
}

inline bool
OpenCLSeperableConv::RunKernel(cl_kernel kernel, cl_command_queue queue,
                               cl_mem input_buffer, cl_mem output_buffer,
                               cl_mem gaussian_kernel, cl_uint width, cl_uint height,
                               cl_uint pitch, cl_uint k_w, cl_uint k_h) {
  cl_int err;

  int arg_index = 0;
  err  = clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), (void*)&input_buffer);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), (void*)&output_buffer);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), (void*)&gaussian_kernel);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_uint), (void*)&width);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_uint), (void*)&height);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_uint), (void*)&pitch);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_uint), (void*)&k_w);
  err |= clSetKernelArg(kernel, arg_index++, sizeof(cl_uint), (void*)&k_h);
  if (err != CL_SUCCESS) {
    std::cerr << "RunKernel failed" << std::endl;
    return false;
  }

  size_t globalWorkSize[2] = { (size_t)height, (size_t)width};
  err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    std::cerr << "clEnqueueNDRangeKernel failed return " << err << std::endl;
  }

  return true;
}


inline bool OpenCLSeperableConv::Run(const cv::Mat& input, const std::vector<float>& kernel, cv::Mat& output) {
  const int width = input.cols;
  const int height = input.rows;
  const int channels = input.channels();
  const size_t image_size = width * height * channels;

  CV_Assert(input.depth() == CV_8U);

  cl_int err = CL_SUCCESS;
  cl_mem input_buf = clCreateBuffer(context_,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    image_size * sizeof(uchar),
    (void*)input.data, &err
  );
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer input_buf failed return " << err << std::endl;
    return false;
  }

  cl_mem output_buf = clCreateBuffer(context_,
    CL_MEM_WRITE_ONLY,
    image_size * sizeof(uchar),
    nullptr, &err
  );
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer output_buf failed return " << err << std::endl;
    return false;
  }

  cl_mem kernel_buf = clCreateBuffer(
    context_,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    kernel.size() * sizeof(float),
    (void*)kernel.data(), &err
  );
  if (err != CL_SUCCESS) {
    std::cerr << "clCreateBuffer kernel_buf failed return " << err << std::endl;
    return false;
  }

  uchar k_w = sqrt(kernel.size());
  uchar k_h = sqrt(kernel.size());
  if (k_w * k_h != kernel.size()) {
    std::cerr << "kernel size incorrect!!!" << k_h << ", " << k_w << ", " << kernel.size() << std::endl;
  }
  RunKernel(
    kernel_, queue_,
    input_buf, output_buf, kernel_buf,
    width, height, width * channels,
    k_h, k_w
  );

  clFinish(queue_);

  std::vector<uchar> output_host(image_size);
  err = clEnqueueReadBuffer(
    queue_,
    output_buf,
    CL_TRUE,
    0,
    image_size * sizeof(uchar), output_host.data(),
    0, nullptr, nullptr
  );

  if (err != CL_SUCCESS) {
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(kernel_buf);
    std::cerr << "clEnqueueReadBuffer failed return " << err << std::endl;
  }

  output = cv::Mat(height, width, CV_8UC3, output_host.data()).clone();
  clReleaseMemObject(input_buf);
  clReleaseMemObject(output_buf);
  clReleaseMemObject(kernel_buf);
  return true;
}

inline bool OpenCLSeperableConv::IsValid() const { return valid_; }

} // namespace kumo