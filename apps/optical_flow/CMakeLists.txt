################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.0)

project(opticalFlow)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

if(DEFINED OPENVX_INCLUDES)
    message(STATUS "OPENVX_INCLUDES: \t${OPENVX_INCLUDES}")
else()
    set(OPENVX_INCLUDES /opt/rocm/mivisionx/include)
    message(STATUS "OPENVX_INCLUDES: \t<empty> \t Default:/opt/rocm/mivisionx/include")
endif()
include_directories(${OPENVX_INCLUDES})

if(DEFINED OPENVX_LIBRARIES)
    message(STATUS "OPENVX_LIBRARIES: \t${OPENVX_LIBRARIES}")
else()
    set(OPENVX_LIBRARIES /opt/rocm/mivisionx/lib/libopenvx.so)
    message(STATUS "OPENVX_LIBRARIES: \t<empty> \t Default:/opt/rocm/mivisionx/lib/libopenvx.so")
endif()

add_executable(${PROJECT_NAME} source/optical_flow.cpp)
target_link_libraries(${PROJECT_NAME} ${OPENVX_LIBRARIES} ${OpenCV_LIBRARIES})
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)