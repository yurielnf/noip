#include "fermionic.h"
#include <carma>
#include <armadillo>
#include <itensor/all.h>

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

/// return the rotation that the circuit apply in practice
arma::mat applyCircuitNO(string filename="psi.txt", int nExcludeLeftSite=6)
{
    itensor::MPS psi;
    itensor::Fermion sites(50, {"ConserveNf=",true});
    auto cc=Fermionic::cc_matrix(psi, sites);
    auto givens=Fermionic::NOGivensRot(cc,nExcludeLeftSite,20);
    auto rot1=matrot_from_Givens(givens,cc.n_rows);
    auto gates=Fermionic::NOGates(sites,givens);
    gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
    return rot1.t();
}



//---------------------------------- start python module ------------------


PYBIND11_MODULE(it_irlm_py, m) {
  m.doc() = "Python interface for it_irlm";

//py::class_<Fermionic>(m,"Fermionic")
//          .def(py::init<arma::mat,arma::mat, std::map<std::array<int,4>,double>>(),
//               "Kij"_a, "Uij"_a, "Vijkl"_a)
//          .def(py::init<arma::mat,arma::mat,arma::mat,bool>(),
//               "Kij"_a, "Uij"_a, "Rot"_a, "rotateKin"_a=true)
//          .def("Ham",&Fermionic::Ham, "tol"_a=1e-14)
//          .def("NParticle",&Fermionic::NParticle)
//          .def("CidCj",&Fermionic::CidCj)
//          ;


  m.def("applyCircuitNO",&applyCircuitNO, "filename"_a="psi.txt", "nExcludeLeftSite"_a=6);

}
