#include "scan_cl.hpp"
#include <benchmark/benchmark.h>
#include <cstring>
#include <glog/logging.h>
#include <cmath>
#include <random>
#include <string>

std::vector<int> generate_input(int length, int min_val = 0, int max_val = 255) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_val, max_val);

  std::vector<int> input(length);
  for (int& val : input) {
    val = dis(gen);
  }
  return input;
}

void ScanHost(const std::vector<int>& input, std::vector<int>& output) {
  output.at(0) = 0;
  for (int i = 1; i < input.size(); i++) {
    output.at(i) = input.at(i) + input.at(i-1);
  }
}

bool is_result_correct(const std::vector<int>& input, const std::vector<int>& output) {
  if (input.size() != output.size()) {
    return false;
  }

  if (output.at(0) != 0) {
    return false;
  }

  for (int i = 1; i < input.size(); i++) {
    if (output.at(i) != input.at(i) + input.at(i-1))
      return false;
  }

  return true;
}

static void BM_PrefixSumGPU(benchmark::State& state) {
  size_t array_length = state.range(0);
  auto input = generate_input(array_length, 0, 255);
  auto output = std::vector<int>(array_length);

  kumo::ScanCL scan_runtime;
  scan_runtime.Init();
  
  for (auto _ : state) {
    scan_runtime.Run(input, output);
    for (int val : output)
      benchmark::DoNotOptimize(val);
  }

  if (!is_result_correct(input, output)) {
    std::cout << "result incorrect!\n";
  }

  state.SetItemsProcessed(state.iterations() * array_length); // Processed 'array_length' elements in each iteration
  state.SetLabel("BM_PrefixSumHost" + std::string("_arraylength_") + std::to_string(array_length));
}


static void BM_PrefixSumHost(benchmark::State& state) {
  size_t array_length = state.range(0);
  auto input = generate_input(array_length, 0, 255);
  auto output = std::vector<int>(array_length);


  for (auto _ : state) {
    ScanHost(input, output);
    for (int val : output)
      benchmark::DoNotOptimize(val);
  }

  if (!is_result_correct(input, output)) {
    std::cout << "result incorrect!\n";
  }

  state.SetItemsProcessed(state.iterations() * array_length); // Processed 'array_length' elements in each iteration
  state.SetLabel("BM_PrefixSumHost" + std::string("_arraylength_") + std::to_string(array_length));
}

BENCHMARK(BM_PrefixSumHost)
  ->Args({1024})
  ->Args({2048})
  ->Args({4096});

BENCHMARK(BM_PrefixSumGPU)
  ->Args({1024})
  ->Args({2048})
  ->Args({4096});

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