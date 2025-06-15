#include <benchmark/benchmark.h>
#include <cstring>
#include <glog/logging.h>
#include "OpenCLRuntime.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <vector>

std::string g_input_path;
std::string g_output_path;

std::vector<float> createGaussianKernel1D(int radius, float sigma) {
  int size = 2 * radius + 1;
  std::vector<float> kernel(size);
  float sum = 0.0f;
  for (int i = -radius; i < radius; ++i) {
    float value = std::exp( -(i * i) / (2.0f * sigma * sigma) );
    kernel[i + radius] = value;
    sum += value;
  }

  for (auto &v : kernel) {
    v /= sum;
  }

  return kernel;
}

void gaussianBlur1D(const cv::Mat& src, cv::Mat& dst, const std::vector<float>& kernel, bool horizontal) {
  CV_Assert(src.channels() == 1 || src.channels() == 3);
  dst = cv::Mat::zeros(src.size(), src.type());
  int radius = static_cast<int>(kernel.size() / 2);

  for (int y = 0; y < src.rows; ++y) {
    for (int x = 0; x < src.cols; ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        float sum = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
          int xx = horizontal ? x + k : x;
          int yy = horizontal ? y : y + k;
          xx = std::clamp(xx, 0, src.cols - 1);
          yy = std::clamp(yy, 0, src.rows - 1);
          if (src.channels() == 1) {
            sum += kernel[k + radius] * src.at<uchar>(yy, xx);
          } else {
            sum += kernel[k + radius] * src.at<cv::Vec3b>(yy, xx)[c];
          }
        }
        if (src.channels() == 1) {
          dst.at<uchar>(y, x) = static_cast<uchar>(sum);
        } else {
          dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(sum);
        }
      }
    }
  }
}

static void BM_GaussianBlur1D(benchmark::State& state) {
  cv::Mat input = cv::imread(g_input_path, cv::IMREAD_COLOR);
  CHECK(!input.empty()) << "Failed to load image!";

  int radius = static_cast<int>(state.range(0));
  float sigma = static_cast<float>(state.range(1)) / 10.0f;
  auto kernel = createGaussianKernel1D(radius, sigma);

  cv::Mat temp, output;
  for (auto _ : state) {
    gaussianBlur1D(input, temp, kernel, true);
    gaussianBlur1D(temp, output, kernel, false);

    benchmark::DoNotOptimize(output.data);
  }

  state.SetItemsProcessed(state.iterations() * input.total());
  state.SetLabel("GaussianBlur1D_" + std::to_string(radius) + "_sigma_" + std::to_string(sigma));

  std::string output_path = g_output_path;
  std::string filename = "_blurred_radius" + std::to_string(radius) +
                         "_sigma" + std::to_string(sigma) + ".png";
  cv::imwrite(output_path + filename, output);
}

BENCHMARK(BM_GaussianBlur1D)
  ->Args({3, 15})
  ->Args({5, 20})
  ->Args({7,25});

int main(int argc, char** argv) {
  // 先初始化Google Benchmark，解析它的参数
  benchmark::Initialize(&argc, argv);

  // 解析还剩下的其他命令行参数
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--input=") == 0) {
      g_input_path = arg.substr(strlen("--input="));
    } else if (arg.find("--output=") == 0) {
      g_output_path = arg.substr(strlen("--output="));
    } else {
      std::cout << "Unknown param: " << arg << std::endl;
    }
  }

  if (g_input_path.empty() || g_output_path.empty()) {
    std::cerr << "Please type: --input=/input/path --output=/output/path parameters!!!\n";
    return 1;
  }

  std::cout << "input path: " << g_input_path << std::endl;
  std::cout << "output path: " << g_output_path << std::endl;

  // 运行所有注册的benchmark
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}