#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main()
{
    int len=10;
    IRLM model {.L=len, .t=0.5, .V=0.1, .U=0.5, .ed=-10};
    HamSys sys=model.HamStar();
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;

    // solve the gs of system
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=128;
    sol_gs.noise=1e-3;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<6; i++) {
        if (i==3) {
            sol_gs.noise=1e-8;
            sol_gs.nIter_diag=32;
        }
        sol_gs.iterate();
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }

    sol_gs.psi.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
     //return 0;

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto sys2=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.5, .ed=0.0}.HamStar();
    cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
    it_tdvp sol {sys2, sol_gs.psi};
    sol.bond_dim=1024;
    sol.noise=1e-8;
    //ofstream out("irlm_star_L"s+to_string(sol.hamsys.ham.length())+".txt");
    cout<<"sweep bond-dim energy n0\n";
    cout<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<" "<<itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real()<<endl;
    cout<<setprecision(10);
    for(auto i=0u; i<len*10/2; i++) {
        if (i==3) sol.noise=0;
        if (i%20==0) {
            auto cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
            arma::eig_sym(cc).print("evals=");
        }
        sol.iterate();
        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        cout<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<" "<<n0<<endl;
    }

    return 0;
}
