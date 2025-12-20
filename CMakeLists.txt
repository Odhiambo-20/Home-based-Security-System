cmake_minimum_required(VERSION 3.15)
project(BiometricSecuritySystem VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${OpenCV_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(firmware)
add_subdirectory(tests)

# Main executable
add_executable(biometric_security
    src/main.cpp
)

target_link_libraries(biometric_security
    core_lib
    biometrics_lib
    ai_ml_lib
    security_lib
    ${OpenCV_LIBS}
    Threads::Threads
    tensorflow-lite
    mbedtls
)

# Install targets
install(TARGETS biometric_security DESTINATION /usr/local/bin)
install(DIRECTORY config/ DESTINATION /etc/biometric_security)
install(DIRECTORY models/ DESTINATION /usr/local/share/biometric_security/models)
