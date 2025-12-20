# Firmware CMakeLists.txt
add_library(firmware_drivers
    drivers/biometric_sensor_driver.cpp
    drivers/camera_driver.cpp
    drivers/display_driver.cpp
    drivers/network_driver.cpp
    drivers/secure_element_driver.cpp
)

target_include_directories(firmware_drivers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
