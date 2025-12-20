add_library(ui_lib
    display_manager.cpp
    ui_renderer.cpp
    touch_handler.cpp
    animation_engine.cpp
)
target_include_directories(ui_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
