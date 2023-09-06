#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

int main()
{
    IRLM model {.L=20, .t=0.5, .V=0.15, .U=0.1};
    HamSys sys=IRLM {.L=20, .t=0.5, .V=0.15, .U=0.1}.HamStar();
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
    auto cc1=model.cc_matrix(sol_gs.psi, sol_gs.hamsys.sites);

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";


    auto rot=model.rotStar();
    itensor::MPS psi=sol_gs.psi;
    {
        IRLM model {.L=20, .t=0.5, .V=0.15, .U=-0.5};
        auto sys2=model.HamStar();
        cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
        it_tdvp sol {sys2, psi};
        sol.bond_dim=256;
        ofstream out("irlm_star_L"s+to_string(sol.hamsys.ham.length())+".txt");
        out<<"sweep bond-dim energy\n"<<setprecision(16);
        out<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        for(auto i=0u; i<100; i++) {
            sol.iterate();
            out<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        }
        auto cc=model.cc_matrix(sol.psi, sol.hamsys.sites);
        rot=model.rotNO(cc);
        psi=sol.psi;
    }

    cout<<"\n-------------------------- rotate the psi to natural orbitals ----------------\n";

    {
        auto sys3=model.rotOp(rot);
        it_tdvp sol {sys3, psi};
        sol.bond_dim=256;
        ofstream out("irlm_nat_orb_L"s+to_string(sol.hamsys.ham.length())+".txt");
        out<<setprecision(16);
        out<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        for(auto i=0u; i<10; i++) {
            sol.iterate();
            out<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        }
    }

    return 0;
}
