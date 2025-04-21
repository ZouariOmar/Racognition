# Set the CMake minimum version
cmake_minimum_required(VERSION 3.31.5)

# Set the project name
project(Racognition VERSION 0.1 LANGUAGES CXX)

# Add Qt's uic for processing .ui files
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Set the CMake C/C++ version
set(CMAKE_CXX_STANDARD 20)

# Add a compiler flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Add source(s) files
set(SOURCES
  ${CMAKE_SOURCE_DIR}/../src/FaceRecognizer.cpp
  ${CMAKE_SOURCE_DIR}/../src/main.cpp
)

# Add header(s) files
set(HEADERS
  ${CMAKE_SOURCE_DIR}/../include/FaceRecognizer.hpp
)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# Find OpenCV package
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets ${OpenCV_LIBS})

# Add the tests subdirectory
# add_subdirectory(test)

# Make this a GUI application on Windows
if(WIN32)
  set(CMAKE_WIN32_EXECUTABLE ON)
endif()
