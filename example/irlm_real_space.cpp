#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main()
{
    IRLM model {.L=30, .t=0.5, .V=0.15, .U=-0.5};
    HamSys sys=model.Ham();
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;

    // solve the gs of system
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=64;
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

    sol_gs.psi.orthogonalize({"Cutoff",1e-10});
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";

    return 0;

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto sys2=IRLM {.L=20, .t=0.5, .V=0.15, .U=-0.5}.Ham();
    cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
    it_tdvp sol {sys2, sol_gs.psi};
    sol.bond_dim=256;
    ofstream out("irlm_real_space_L"s+to_string(sol.hamsys.ham.length())+".txt");
    out<<"sweep bond-dim energy\n";
    out<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
    out<<setprecision(16);
    for(auto i=0u; i<100; i++) {
        sol.iterate();
        out<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
    }

    return 0;
}
