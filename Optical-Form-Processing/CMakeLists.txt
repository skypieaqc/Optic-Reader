cmake_minimum_required(VERSION 3.16)
project(omr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Users/guts/Desktop/opencv/build") 

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/core
)

add_executable(omr
    src/main.cpp
    src/core/BubbleDetector.cpp
    src/core/ROIDetector.cpp
    src/core/PerspectiveCorrector.cpp    
    src/core/AnswerKey.cpp            
    src/core/CornerFinder.cpp         
)

target_link_libraries(omr ${OpenCV_LIBS})

if(WIN32)
    set_target_properties(omr PROPERTIES LINK_FLAGS "/SUBSYSTEM:CONSOLE")
endif()