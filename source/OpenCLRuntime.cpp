#include "OpenCLRuntime.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <cstddef>
#include <fstream>
#include <glog/logging.h>
#include <iterator>
#include <vector>

namespace kumo {

OpenCLRuntime::OpenCLRuntime()
  : platform_(nullptr), device_(nullptr), context_(nullptr),
    queue_(nullptr), program_(nullptr), kernel_(nullptr) {}

OpenCLRuntime::~OpenCLRuntime() {
  if (kernel_) clReleaseKernel(kernel_);
  if (program_) clReleaseProgram(program_);
  if (queue_) clReleaseCommandQueue(queue_);
  if (context_) clReleaseContext(context_);
}

bool OpenCLRuntime::init() {
  cl_int err;

  // get platfrom
  cl_uint num_platforms = 0;
  err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to get OpenCL platform IDs.\n";
    return false;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to get platform IDs.\n";
    return false;
  }
  platform_ = platforms[0];

  // get device
  cl_uint num_devices = 0;
  err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (err != CL_SUCCESS || num_devices == 0) {
    LOG(ERROR) << "Failed to find any GPU device.\n";
    return false;
  }
  std::vector<cl_device_id> devices(num_devices);
  err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to get OpenCL device IDs.\n";
    return false;
  }
  device_ = devices[0];

  // create context
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (!context_ || err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create OpenCL context.\n";
    return false;
  }

  // create command queue
#if CL_TARGET_OPENCL_VERSION >= 200
  queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
#else
  queue_ = clCreateCommandQueue(context_, device_, 0, &err);
#endif
  if (!queue_ || err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create command queue.\n";
    return false;
  }

  return true;
}

bool OpenCLRuntime::buildKernelFromFile(const std::string& file_path, const std::string& kernel_name) {
  cl_int err;

  // read kernel code
  std::ifstream file(file_path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open kernel file: " << file_path << "\n";
    return false;
  }
  std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  const char* source_cstr = source.c_str();
  size_t source_size = source.size();

  if (program_) {
    clReleaseProgram(program_);
    program_ = nullptr;
  }

  if (kernel_) {
    clReleaseKernel(kernel_);
    kernel_ = nullptr;
  }

  // create program
  program_ = clCreateProgramWithSource(context_, 1, &source_cstr, &source_size, &err);
  if (!program_ || err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create CL program from source.\n";
    return false;
  }

  // compile program
  err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    // get build log
    size_t log_size = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    LOG(ERROR) << "Error building program:\n" << log.data() << "\n"; 
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  // create kernel
  kernel_ = clCreateKernel(program_, kernel_name.c_str(), &err);
  if (!kernel_ || err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create kernel: " << kernel_name << "\n";
    clReleaseProgram(program_);
    program_ = nullptr;
    return false;
  }

  return true;
}

cl_kernel OpenCLRuntime::getKernel() const {
  return kernel_;
}

cl_mem OpenCLRuntime::createBuffer(size_t size, cl_mem_flags flags, void* host_ptr) {
  cl_int err;
  cl_mem buf = clCreateBuffer(context_, flags, size, host_ptr, &err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create buffer.\n";
    return nullptr;
  }
  return buf;
}

void OpenCLRuntime::writeBuffer(cl_mem buf, const void* data, size_t size) {
  cl_int err = clEnqueueWriteBuffer(queue_, buf, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to write buffer.\n";
  }
  clFinish(queue_);
}

void OpenCLRuntime::readBuffer(cl_mem buf, void* data, size_t size) {
  cl_int err = clEnqueueReadBuffer(queue_, buf, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to read buffer.\n";
  }
  clFinish(queue_);
}

void OpenCLRuntime::runKernel(const std::vector<size_t>& global, const std::vector<size_t>& local) {
  CHECK(kernel_ != nullptr) << "cl_kernel is null\n";
  cl_int err = clEnqueueNDRangeKernel(queue_, kernel_, (cl_uint)global.size(), nullptr, global.data(), local.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to enqueue kernel.\n";
  }
  clFinish(queue_);
}

void OpenCLRuntime::setKernelArg(cl_uint idx, size_t size, const void* value) {
  cl_int err = clSetKernelArg(kernel_, idx, size, value);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to set kernel argument " << idx << ".\n";
  }
}

}