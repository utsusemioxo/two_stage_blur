find_package(benchmark REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(OpenCLTest main.cpp)
target_link_libraries(OpenCLTest
    PRIVATE
    benchmark::benchmark
    glog::glog
    ${OpenCV_LIBS}
    OpenCLRuntime
)