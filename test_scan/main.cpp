#include "ScanCL.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <glog/logging.h>
#include <cmath>
#include <string>

void ScanHost(const std::vector<int>& input, std::vector<int> output) {
  output.at(0) = 0;
  for (int i = 1; i < input.size(); i++) {
    output.at(i) = input.at(i) + input.at(i-1);
  }
}

int main(int argc, char** argv) {
  // 先初始化Google Benchmark，解析它的参数
  benchmark::Initialize(&argc, argv);

  // 解析还剩下的其他命令行参数
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    // if (arg.find("--input=") == 0) {
    //   // 
    // } else if (arg.find("--output=") == 0) {
    //   //
    // } else {
    //   std::cout << "Unknown param: " << arg << std::endl;
    // }
  }


  // 运行所有注册的benchmark
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}