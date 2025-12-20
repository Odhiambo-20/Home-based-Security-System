add_library(security_lib
    encryption_engine.cpp
    secure_storage.cpp
    authentication.cpp
    intrusion_detection.cpp
    secure_boot.cpp
)
target_include_directories(security_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
