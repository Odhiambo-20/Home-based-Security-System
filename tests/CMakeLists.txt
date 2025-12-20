# Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Unit tests
add_executable(unit_tests
    unit_tests/test_biometrics.cpp
    unit_tests/test_security.cpp
    unit_tests/test_ai_ml.cpp
    unit_tests/test_hardware.cpp
)
target_link_libraries(unit_tests ${GTEST_LIBRARIES} pthread)

# Integration tests
add_executable(integration_tests
    integration_tests/test_full_system.cpp
    integration_tests/test_performance.cpp
)
target_link_libraries(integration_tests ${GTEST_LIBRARIES} pthread)

enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)
