# AI/ML Models Directory

This directory contains the trained neural network models for biometric authentication.

## Models

1. **fingerprint_cnn.tflite** - Fingerprint recognition CNN (TensorFlow Lite)
2. **palm_recognition.onnx** - Palm print recognition network (ONNX)
3. **liveness_detection.tflite** - Anti-spoofing liveness detection (TensorFlow Lite)
4. **anomaly_detection.onnx** - Behavioral anomaly detection (ONNX)

## Training

Models should be trained using the Python scripts in the `scripts/` directory.
See training documentation for details.

## Model Updates

Models can be updated via OTA (Over-The-Air) updates using the deployment system.
