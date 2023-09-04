#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include<iostream>
#include<iomanip>

using namespace std;

int main()
{
    IRLM model {.L=20, .t=0.5, .V=0.15, .U=0.1};
    HamSys sys=model.HamStar();
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

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto rot=model.rotStar();
    auto sys2=IRLM {.L=20, .t=0.5, .V=0.15, .U=-0.5}.HamStar();
    cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
    it_tdvp sol {sys2, sol_gs.psi};
    sol.bond_dim=256;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<100; i++) {
        sol.iterate();
        cout<<i+1<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
    }

    return 0;
}
