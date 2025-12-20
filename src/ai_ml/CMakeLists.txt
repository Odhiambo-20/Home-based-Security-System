add_library(ai_ml_lib
    neural_network.cpp
    feature_extractor.cpp
    liveness_detection.cpp
    anomaly_detector.cpp
    model_optimizer.cpp
)
target_include_directories(ai_ml_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
