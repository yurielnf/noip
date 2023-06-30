pybind11_add_module(${PROJECT_NAME}py
    bindings.cpp
)

target_link_libraries(${PROJECT_NAME}py PUBLIC ${PROJECT_NAME})
