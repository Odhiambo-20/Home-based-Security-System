# Third-Party Dependencies

This directory contains external libraries used by the Biometric Security System.

## Required Libraries

1. **TensorFlow Lite** - Deep learning inference
2. **OpenCV** - Computer vision algorithms
3. **mbedTLS** - Cryptographic functions
4. **FreeRTOS** - Real-time operating system (optional)
5. **Eigen** - Linear algebra

## Installation

Install system-wide dependencies:
```bash
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev \
    libtensorflow-lite-dev \
    libmbedtls-dev \
    libeigen3-dev
```

Or place library sources in this directory for static linking.
