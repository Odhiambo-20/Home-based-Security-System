#!/bin/bash
# Build script for Biometric Security System

set -e

echo "Building Biometric Security System..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build with all CPU cores
make -j$(nproc)

echo "Build complete!"
echo "Binary location: build/biometric_security"
