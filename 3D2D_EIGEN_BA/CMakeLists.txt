cmake_minimum_required( VERSION 2.8 )

project( BA )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-o3")

Find_Package(Eigen3 REQUIRED)
Find_Package(OpenCV REQUIRED)
Find_Package(Sophus REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS}) 

add_executable( BA eigen_ba.cpp)

target_link_libraries(BA ${G2O_LIBS} ${CHOLMOD_LIBRARIES} ${OpenCV_LIBS} ${Sophus_LIBRARIES})