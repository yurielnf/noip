include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        master
)

FetchContent_Declare(
  armadillo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG        11.4.x
)

#FetchContent_Declare(
#  carma
#  GIT_REPOSITORY https://github.com/RUrlus/carma.git
#  GIT_TAG        stable
#)

FetchContent_MakeAvailable(pybind11 armadillo)


add_library(itensor STATIC IMPORTED) # or STATIC instead of SHARED
set_target_properties(itensor PROPERTIES
  IMPORTED_LOCATION_RELEASE "/home/yurielnf/opt/ITensor/lib/libitensor.a"
  IMPORTED_LOCATION_DEBUG "/home/yurielnf/opt/ITensor/lib/libitensor-g.a"
  INTERFACE_INCLUDE_DIRECTORIES "/home/yurielnf/opt/ITensor"
)

add_library(tdvp INTERFACE IMPORTED) # or STATIC instead of SHARED
set_target_properties(tdvp PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/home/yurielnf/opt/TDVP"
)


find_package(OpenMP REQUIRED)
