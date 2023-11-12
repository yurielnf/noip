cmake_minimum_required(VERSION 3.9)

project(it_tdvp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external/external.cmake)

add_library(${PROJECT_NAME} STATIC
    fermionic.h
    irlm.h
    it_dmrg.h
    it_tdvp.h
    empty.cpp
    irlm_wilson.h
    it_tdvp_error.h
    irlm_wilson_chain.h
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${MKL_INCLUDE_DIRS}
)

#set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)



target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,--start-group" ${MKL_SEQUENTIAL_LAYER_LIBRARY} "-Wl,--end-group -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)

target_link_libraries(${PROJECT_NAME} PUBLIC armadillo)

##------------ example ----------

add_subdirectory(example)

##------------ wrapper ----------

add_subdirectory(python)
