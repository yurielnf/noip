#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

void rotateToNO(itensor::MPS psi, arma::mat const& rot)
{
    auto sys3=Fermionic::rotOp(rot);
    it_tdvp sol {sys3, psi};
    sol.dt={0,0.005};
//        sol.err_goal=1e-12;
    sol.bond_dim=256;
    ofstream out("irlm_nat_orb_L"s+to_string(sol.hamsys.ham.length())+".txt");
    out<<setprecision(16);
    sol.psi.orthogonalize({"Cutoff",1e-7});
    for(auto i=0; i<sol.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
    cout<<endl;
    out<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
    for(auto i=0u; i*sol.dt.imag()<1.0; i++) {
        sol.iterate();
        if ((i+1)*sol.dt.imag()>=1.0) {
            sol.psi.orthogonalize({"Cutoff",1e-7});
            for(auto i=0; i<sol.psi.length(); i++)
                cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
            cout<<endl;
        }
        out<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
    }
    auto cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
    cc.diag().print("ni");
}


int main()
{
    int len=20;
    IRLM model {.L=len, .t=0.5, .V=0.15, .U=-0.5};
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
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    auto cc1=arma::real( Fermionic::cc_matrix(sol_gs.psi, sol_gs.hamsys.sites) );

    cout<<"\n-------------------------- rotate the psi to natural orbitals ----------------\n";

    rotateToNO(sol_gs.psi, Fermionic::rotNO(cc1));
    return 0;


    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";


    auto rot=model.rotStar();
    itensor::MPS psi=sol_gs.psi;
    {
        IRLM model {.L=len, .t=0.5, .V=0.15, .U=-0.5};
        auto sys2=model.HamStar();
        cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
        it_tdvp sol {sys2, psi};
        sol.bond_dim=256;
        ofstream out("irlm_star_L"s+to_string(sol.hamsys.ham.length())+".txt");
        out<<"sweep bond_dim energy\n"<<setprecision(16);
        out<<"0 "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        for(auto i=0u; i<100; i++) {
            sol.iterate();
            out<<(i+1)*abs(sol.dt)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        }
        arma::mat cc=arma::real(Fermionic::cc_matrix(sol.psi, sol.hamsys.sites));
        cc.diag().print("ni");
        rot=Fermionic::rotNO(cc);//model.rotStar();//
        psi=sol.psi;
    }

    cout<<"\n-------------------------- rotate the psi to natural orbitals ----------------\n";

    rotateToNO(psi, rot);

    return 0;
}
