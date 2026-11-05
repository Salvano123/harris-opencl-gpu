cmake_minimum_required(VERSION 3.15)
project(HarrisVertexDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED) # Tylko do wczytywania/zapisu obrazów i wizualizacji

include_directories(include ${OpenCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(harris_app 
    src/main.cpp 
    src/HarrisDetectorCL.cpp
)

target_link_libraries(harris_app PRIVATE ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

# Automatyczne kopiowanie folderu z kernelami do katalogu budowania
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels DESTINATION ${CMAKE_CURRENT_BINARY_DIR})