cmake_minimum_required ( VERSION 2.6 )

# GCC is also in Courseware...
#SET ( CMAKE_CXX_COMPILER "g++-5.2.0" )
#SET ( CMAKE_C_COMPILER "gcc-5.2.0" )


## ---------------------------------------- START PROJECT ---------------------------------------- ##
project ( eoa_image_compression )

#SET ( CMAKE_VERBOSE_MAKEFILE ON )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set ( CMAKE_CXX_FLAGS "-g -std=c++11 -O3" )
#set ( CMAKE_CXX_FLAGS "-g -std=c++11" )
set ( DEBUG_MODE ON )

find_package ( CUDA )
list ( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35;-std=c++11;-DVERBOSE" )
set ( CUDA_PROPAGATE_HOST_FLAGS OFF )


#add_definitions ( -DMEASURE_TIME )


FIND_PACKAGE ( OpenCV 3.1 REQUIRED )
IF ( OpenCV_FOUND )
    MESSAGE ( STATUS "OpenCV was found." )
ELSE ( OpenCV_FOUND )
    MESSAGE ( FATAL_ERROR "OpenCV was not found." )
ENDIF ( OpenCV_FOUND )


INCLUDE_DIRECTORIES ( src
                      ${OpenCV_INCLUDE_DIR} )

file ( GLOB MY_HEADERS src/*.h
                       src/shapes/*.h
                       src/components/*.h
                       src/algorithms/*.h
                       src/algorithms/entities/*.h
                       src/components/fitness/*.h
                       src/components/fitness/cpu/*.h
                       src/components/fitness/gpu/*.h )
file ( GLOB MY_SOURCES src/*.cpp
                       src/shapes/*.cpp
                       src/components/*.cpp
                       src/algorithms/*.cpp
                       src/algorithms/entities/*.cpp
                       src/components/fitness/cpu/*.cpp )
file ( GLOB CU_SOURCES src/components/fitness/gpu/*.cu )

## ----------------------------------------- EXECUTABLES ----------------------------------------- ##

add_executable ( compress
    ${MY_HEADERS}
    ${MY_SOURCES}
)

target_link_libraries ( compress
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)



add_executable ( compress_average
    ${MY_HEADERS}
    ${MY_SOURCES}
)

target_compile_definitions ( compress_average PUBLIC "-DRENDER_AVERAGE" )

target_link_libraries ( compress_average
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)



cuda_add_executable ( compress_gpu
    ${MY_HEADERS}
    ${MY_SOURCES}
    ${CU_SOURCES}
)

target_compile_definitions ( compress_gpu PUBLIC "-DUSE_GPU" )

target_link_libraries ( compress_gpu
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)



cuda_add_executable ( compress_average_gpu
    ${MY_HEADERS}
    ${MY_SOURCES}
    ${CU_SOURCES}
)

target_compile_definitions ( compress_average_gpu PUBLIC "-DRENDER_AVERAGE -DUSE_GPU" )

target_link_libraries ( compress_average_gpu
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)
