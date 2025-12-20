add_library(communication_lib
    mqtt_client.cpp
    ble_manager.cpp
    wifi_manager.cpp
    protocol_handler.cpp
)
target_include_directories(communication_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
