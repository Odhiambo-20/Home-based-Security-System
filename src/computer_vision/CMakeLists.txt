add_library(cv_lib
    image_preprocessor.cpp
    edge_detection.cpp
    feature_matching.cpp
    quality_assessment.cpp
)
target_link_libraries(cv_lib ${OpenCV_LIBS})
target_include_directories(cv_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
