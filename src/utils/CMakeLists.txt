add_library(utils_lib
    logger.cpp
    config_manager.cpp
    time_sync.cpp
    diagnostics.cpp
)
target_include_directories(utils_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
