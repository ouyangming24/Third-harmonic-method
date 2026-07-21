cmake_minimum_required(VERSION 3.10)
project(FiberThermalModel VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找 Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 库文件
add_library(fiber_model STATIC
    fiber_model.cpp
    fiber_model.h
)

target_link_libraries(fiber_model PUBLIC Eigen3::Eigen)

target_include_directories(fiber_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 测试程序
add_executable(test_fiber_model test_fiber_model.cpp)
target_link_libraries(test_fiber_model fiber_model)

# 安装规则
install(TARGETS fiber_model
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES fiber_model.h
    DESTINATION include
)

# 启用测试
enable_testing()
add_test(NAME FiberModelTest COMMAND test_fiber_model)

# 打印配置信息
message(STATUS "FiberThermalModel Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen3 Version: ${Eigen3_VERSION}")



