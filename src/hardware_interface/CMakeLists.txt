add_library(hardware_interface_lib
    sensor_controller.cpp
    actuator_controller.cpp
    display_controller.cpp
    gpio_manager.cpp
)
target_include_directories(hardware_interface_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
