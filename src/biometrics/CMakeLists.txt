add_library(biometrics_lib
    fingerprint_scanner.cpp
    palm_print_analyzer.cpp
    vein_pattern_detector.cpp
    multimodal_fusion.cpp
)
target_include_directories(biometrics_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
