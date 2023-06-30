#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "it_dmrg.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;




//---------------------------------- start python module ------------------


PYBIND11_MODULE(varttpy, m) {
  m.doc() = "Python interface for vartt";

//py::class_<Fermionic>(m,"Fermionic")
//          .def(py::init<arma::mat,arma::mat, std::map<std::array<int,4>,double>>(),
//               "Kij"_a, "Uij"_a, "Vijkl"_a)
//          .def(py::init<arma::mat,arma::mat,arma::mat,bool>(),
//               "Kij"_a, "Uij"_a, "Rot"_a, "rotateKin"_a=true)
//          .def("Ham",&Fermionic::Ham, "tol"_a=1e-14)
//          .def("NParticle",&Fermionic::NParticle)
//          .def("CidCj",&Fermionic::CidCj)
//          ;



}
