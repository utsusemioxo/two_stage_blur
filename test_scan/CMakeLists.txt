find_package(benchmark REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(test_scan main.cpp)
target_link_libraries(test_scan
    PRIVATE
    benchmark::benchmark
    OpenCL::OpenCL
    glog::glog
)