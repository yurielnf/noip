#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

struct State {
    itensor::MPS psi;
    itensor::Fermion sites;
};

State rotateState(itensor::MPS psi, arma::mat const& rot)
{
    auto sys3=Fermionic::rotOp(rot);
    it_tdvp sol {sys3, psi};
    sol.dt={0,0.01};
//        sol.err_goal=1e-12;
    sol.bond_dim=128;
    sol.psi.orthogonalize({"Cutoff",1e-7});
    for(auto i=0; i<sol.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
    cout<<endl;
    for(auto i=0u; i*sol.dt.imag()<1.0; i++) {
        sol.iterate();
        if ((i+1)*sol.dt.imag()>=1.0) {
            sol.psi.orthogonalize({"Cutoff",1e-7});
            for(auto i=0; i<sol.psi.length(); i++)
                cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
            cout<<endl;
        }
        //cout<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys3.ham) <<" "<<maxLinkDim(sol.psi)<<endl;
    }
    auto cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
    cc.diag().print("ni");
    return {sol.psi, sol.hamsys.sites};
}

State computeGS(HamSys sys)
{
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-5;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<10; i++) {
        if (i==3) {
            sol_gs.noise=1e-10;
//            sol_gs.nIter_diag=2;
        }
        else if (i==9) sol_gs.noise=0;
        sol_gs.iterate();
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return {sol_gs.psi, sol_gs.hamsys.sites};
}


int main()
{
    int len=20;

    cout<<"\n-------------------------- solve the gs of system ----------------\n";

    auto model1=IRLM {.L=len, .t=0.5, .V=0.15, .U=0.1};
    auto rot=model1.rotStar();
    auto sol1a=computeGS(model1.Ham(rot, true));


    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    auto cc=Fermionic::cc_matrix(sol1a.psi, sol1a.sites);
    cc.diag().print("ni");
    rot = rot*Fermionic::rotNO(cc);
    auto sol1b=computeGS(model1.Ham(rot));

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto model2=IRLM {.L=len, .t=0.5, .V=0.15, .U=-0.5};
    auto psi=sol1b.psi;

    cout<<"time M m energy\n" << setprecision(12);
    for(auto i=0; i<10; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        auto sys2=model2.Ham(rot);
        it_tdvp sol {sys2, psi};
        sol.bond_dim=256;
        cout<<"0 "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        for(auto k=0; k<1; k++) {
            sol.iterate();
            cout<<(i*1+k+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        }
        cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
        cc.diag().print("ni");
        auto rot1=Fermionic::rotNO(cc);
        psi=rotateState(sol.psi, rot1).psi;
        rot = rot*rot1;
    }

    return 0;
}
