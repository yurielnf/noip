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

State rotateState(itensor::MPS psi, arma::mat const& rot, int nExclude)
{
    auto sys3=Fermionic::rotOp(rot, nExclude);
    it_tdvp sol {sys3, psi};
    sol.dt={0,1};
    sol.bond_dim=256;
    sol.rho_cutoff=1e-14;
    sol.silent=true;
    sol.epsilonM=0e-4;
//    sol.psi.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<sol.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
    cout<<endl;
    for(auto i=0u; i*sol.dt.imag()<1.0; i++) {
        sol.iterate();
        if (i==2) sol.epsilonM=0;
        if ((i+1)*sol.dt.imag()>=1.0) {
            sol.psi.orthogonalize({"Cutoff",1e-9});
            for(auto i=0; i<sol.psi.length(); i++)
                cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
            cout<<endl;
        }
        cout<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys3.ham) <<" "<<maxLinkDim(sol.psi)<<endl;
    }
//    auto cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
//    cc.diag().print("ni");
    return {sol.psi, sol.hamsys.sites};
}

State rotateState2(itensor::MPS psi2, arma::mat const& rot, double dt=0.01)
{
    using namespace itensor;
    auto sys=Fermionic::rotOpExact(rot);
    psi2.replaceSiteInds(sys.sites.inds());
    println("-------------------------------------MPO W^I 2nd order---------------------------------------");
    auto expH1 = toExpH(sys.ampo,(1-1_i)/2*dt*1_i);
    auto expH2 = toExpH(sys.ampo,(1+1_i)/2*dt*1_i);
    printfln("Maximum bond dimension of expH1 is %d",maxLinkDim(expH1));
    auto args = Args("Method=","DensityMatrix","Cutoff=",1E-12,"MaxDim=",2000);
    for(auto i=0u; i*dt<1.0; i++)
    {
        psi2 = applyMPO(expH1,psi2,args);
        psi2.noPrime();
        psi2 = applyMPO(expH2,psi2,args);
        psi2.noPrime().normalize();
    }
    for(auto i=0; i<psi2.length(); i++)
        cout<<itensor::leftLinkIndex(psi2,i+1).dim()<<" ";
    cout<<endl;

    return {psi2, sys.sites};
}

State rotateState3(itensor::MPS psi, arma::mat const& rot, int nExclude=2)
{
    arma::mat rott=rot.t();

    auto [eval,evec]=eig_unitary(rott,nExclude);

//    auto im=std::complex(0.,1.);
//    arma::cx_mat(evec*diagmat(arma::log(eval)*im)*evec.t()).clean(1e-5).print("rotHam");

    itensor::Fermion sites(psi.length(), {"ConserveNf=",false});
    psi.replaceSiteInds(sites.inds());
    cout<<"it m(psi2) m(psi)\n";
    for(auto a=psi.length()-1; a>=nExclude; a--) {
        if (std::abs(eval(a)-1.0)<1e-14) continue;
        itensor::AutoMPO ampo(sites);
        for(auto i=nExclude; i<psi.length(); i++)
            for(auto j=nExclude; j<psi.length(); j++)
                ampo += evec(i,a)*std::conj(evec(j,a)),"Cdag",i+1,"C",j+1;
        auto ha=itensor::toMPO(ampo,{"Cutoff",1e-14});
        if (itensor::maxLinkDim(ha)>4) cout<<"no bond dim 4 in mpo\n";
        auto psi2=itensor::applyMPO(ha, psi, {"Cutoff",1e-10});
        psi2.noPrime();
        if (itensor::norm(psi2)*std::abs(eval(a)-1.0)>1e-14) {
            psi=itensor::sum(psi, psi2*(eval(a)-1.0));
            psi.noPrime();
            psi.orthogonalize({"Cutoff",1e-9});
        }
        cout<<a<<" "<<itensor::maxLinkDim(psi2)<<" "<<itensor::maxLinkDim(psi)<<"\n";
        cout.flush();
    }

    return {psi, sites};
}

auto computeGS(HamSys const& sys)
{
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-5;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<4; i++) {
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
    return sol_gs;
}


/// ./irlm_star <len>
int main(int argc, char **argv)
{
    int len=20, nExclude=2;
    if (argc==2) len=atoi(argv[1]);
    cout<<"\n-------------------------- solve the gs of system ----------------\n";

    auto model1=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=-10};
    auto rot=model1.rotStar();
    //eig_unitary(rot);
    //return 0;
    auto sol1a=computeGS(model1.Ham(rot, true));


    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    auto cc=Fermionic::cc_matrix(sol1a.psi, sol1a.hamsys.sites);
    cc.diag().print("ni");
    rot = rot*Fermionic::rotNO3(cc,nExclude);
    auto sys1b=model1.Ham(rot, nExclude==2);
    auto sol1b=computeGS(sys1b);
    cc=Fermionic::cc_matrix(sol1b.psi, sol1b.hamsys.sites);
    cc.diag().print("ni");


    auto psi1=sol1b.psi;
    psi1.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<psi1.length(); i++)
        cout<<itensor::leftLinkIndex(psi1,i+1).dim()<<" ";
    cout << "\n";

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto model2=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=0.0};

    ofstream out("irlm_no_L"s+to_string(len)+".txt");
    out<<"time M m energy n0\n" << setprecision(12);
    double n0=itensor::expectC(sol1b.psi, sol1b.hamsys.sites, "N",{1}).at(0).real();
    out<<"0 "<< maxLinkDim(sys1b.ham) <<" "<<maxLinkDim(sol1b.psi)<<" "<<sol1b.energy<<" "<<n0<<endl;
    auto psi=sol1b.psi;
    for(auto i=0; i<len*10/2; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        auto sys2=model2.Ham(rot, nExclude==2);
        it_tdvp sol {sys2, psi};
        sol.dt={0,0.1};
        sol.bond_dim=256;
        sol.rho_cutoff=1e-14;
        sol.silent=false;
        sol.epsilonM=(i%10==0) ? 1e-4 : 0;

        sol.iterate();

        psi=sol.psi;
        //psi.orthogonalize({"Cutoff",1e-9});
        cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
        cc.diag().print("ni");
        //double n0=arma::cdot(rot.row(0), cc*rot.row(0).st());
        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        if (i>=0) {
            auto rot1=Fermionic::rotNO3(cc,nExclude);
            out<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<" "<<n0<<endl;
            psi=rotateState3(psi, rot1, nExclude).psi;
            psi.orthogonalize({"Cutoff",1e-9});
            rot = rot*rot1;
        }
        for(auto i=0; i<psi.length(); i++)
            cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
        cout<<endl;
    }

    return 0;
}
