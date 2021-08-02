get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME}_bin main.cpp)
target_include_directories (${PROJECT_NAME}_bin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})