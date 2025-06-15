#pragma once
#include <CL/cl.h>
#include <cstddef>
#include <string>
#include <vector>

namespace kumo {

class OpenCLRuntime {
public:
  OpenCLRuntime();
  ~OpenCLRuntime();

  bool init();
  bool buildKernelFromFile(const std::string& file_path, const std::string& kernel_name);
  cl_kernel getKernel() const;

  cl_mem createBuffer(size_t size, cl_mem_flags flags, void* host_ptr = nullptr);
  void writeBuffer(cl_mem buf, const void* data, size_t size);
  void readBuffer(cl_mem buf, void* data, size_t size);
  void runKernel(const std::vector<size_t>& global, const std::vector<size_t>& local);
  void setKernelArg(cl_uint idx, size_t size, const void* value);


private:
  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
  cl_program program_;
  cl_kernel kernel_;
};

}
