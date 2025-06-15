find_package(OpenCL REQUIRED)
find_package(glog REQUIRED)

add_library(OpenCLRuntime
    STATIC
    OpenCLRuntime.cpp
)

target_include_directories(OpenCLRuntime
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(OpenCLRuntime
    PUBLIC
    OpenCL::OpenCL
    glog::glog
)
