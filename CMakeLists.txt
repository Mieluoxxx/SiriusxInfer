cmake_minimum_required(VERSION 3.20)

# 设置编译命令导出
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 显示详细的构建命令，提供更多的调试信息
set(CMAKE_VERBOSE_MAKEFILE ON)
# 设置构建类型
set(CMAKE_BUILD_TYPE "Debug")
# 设置输出路径
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(siriusx-infer LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(siriusx-infer/src/tensor DIR_TENSOR)

add_library(infer STATIC ${DIR_TENSOR})

# 设置库的链接依赖
target_link_libraries(infer PUBLIC
    GTest::gtest_main
    GTest::gtest
    glog::glog
    armadillo
    pthread
)

# 设置库的包含目录
target_include_directories(infer PUBLIC
    ${glog_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/siriusx-infer/include
    ${Armadillo_INCLUDE_DIR}
)

add_subdirectory(test)
