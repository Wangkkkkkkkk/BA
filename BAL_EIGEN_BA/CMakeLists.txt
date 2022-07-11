cmake_minimum_required( VERSION 2.8 )

project( BA )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-o3")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

Find_Package(Eigen3 REQUIRED)
Find_Package(Sophus REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${EIGEN3_INCLUDE_DIR} ${Sophus_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/ceres) 
include_directories(${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/tools 
                    ${PROJECT_SOURCE_DIR}/common/flags)

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

add_executable( BA eigen_ba.cpp)

target_link_libraries(BA ${G2O_LIBS} ${Sophus_LIBRARIES} BALProblem ParseCmd)