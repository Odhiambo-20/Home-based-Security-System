add_library(core_lib
    system_manager.cpp
    security_manager.cpp
    power_manager.cpp
)
target_include_directories(core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
