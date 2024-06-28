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

FetchContent_Declare(
 carma
 GIT_REPOSITORY https://github.com/RUrlus/carma.git
 GIT_TAG        stable
)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.x
)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)

FetchContent_MakeAvailable(pybind11 armadillo carma Catch2 json)


add_library(itensor STATIC IMPORTED) # or STATIC instead of SHARED
set_target_properties(itensor PROPERTIES
  IMPORTED_LOCATION "$ENV{HOME}/opt/ITensor/lib/libitensor.a"
  IMPORTED_LOCATION_RELEASE "$ENV{HOME}/opt/ITensor/lib/libitensor.a"
  IMPORTED_LOCATION_DEBUG "$ENV{HOME}/opt/ITensor/lib/libitensor-g.a"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/opt/ITensor"
)

add_library(tdvp INTERFACE IMPORTED) # or STATIC instead of SHARED
set_target_properties(tdvp PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/opt/TDVP"
)


find_package(OpenMP REQUIRED)

include(external/FindMKL.cmake)
