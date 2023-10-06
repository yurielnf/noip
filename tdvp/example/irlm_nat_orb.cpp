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
    sol.err_goal=1e-7;
//    sol.noise=1e-10;
    sol.bond_dim=128;
//    sol.psi.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<sol.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol.psi,i+1).dim()<<" ";
    cout<<endl;
    for(auto i=0u; i*sol.dt.imag()<1.0; i++) {
        sol.iterate();
        if (i==2) sol.epsilonM=0;
        if ((i+1)*sol.dt.imag()>=1.0) {
//            sol.psi.orthogonalize({"Cutoff",1e-9});
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

State rotateState3(itensor::MPS psi, arma::mat const& rot)
{
//    const auto im=arma::cx_double(0,1);
    arma::mat rott=rot.t();

    auto [eval,evec]=eig_unitary(rott);
//    {// checking quality
//        arma::cx_mat logrot;//=arma::logmat(rott);
//        {
//            arma::cx_mat evec;
//            arma::cx_colvec eval;
//            arma::eig_gen(eval,evec,rott);
//            logrot=evec*arma::diagmat(arma::log(eval))*evec.i();
//        }
//        double err=arma::norm(rott-arma::expmat(logrot));
//        if (err>1e-13) cout<<"exp error="<<err<<endl;
//        double err_herm=norm(logrot.t()+logrot);
//        if (err_herm>1e-13) cout<<"aHermitian error="<<err_herm<<endl;

//        arma::vec eval1;
//        arma::eig_sym(eval1,evec, logrot*im);
//        eval=arma::exp(eval1*(-im));

//        double err2=arma::norm(rott-evec*arma::diagmat(eval)*evec.t());
//        if (err2>1e-13) cout<<"eigen error="<<err2<<endl;

//        cout.flush();
//    }
    itensor::Fermion sites(psi.length());
    psi.replaceSiteInds(sites.inds());
    cout<<"it m(psi2) m(psi)\n";
    for(auto a=0u; a<psi.length(); a++) {
        itensor::AutoMPO ampo(sites);
        for(auto i=0; i<psi.length(); i++)
            for(auto j=0; j<psi.length(); j++)
                ampo += std::conj(evec(j,a))*evec(i,a),"Cdag",i+1,"C",j+1;
        auto ha=itensor::toMPO(ampo);
        if (itensor::maxLinkDim(ha)>4) cout<<"no bond dim 4 in mpo\n";
        auto psi2=itensor::applyMPO(ha, psi);
        psi2.noPrime();
        psi=itensor::sum(psi, psi2*(eval(a)-1.0));
        psi.noPrime();
        cout<<a<<" "<<itensor::maxLinkDim(psi2)<<" "<<itensor::maxLinkDim(psi)<<"\n";
    }

    return {psi, sites};
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
    int len=30;

    cout<<"\n-------------------------- solve the gs of system ----------------\n";

    auto model1=IRLM {.L=len, .t=0.5, .V=0.15, .U=-0.5};
    auto rot=model1.rotStar();
    //eig_unitary(rot);
    //return 0;
    auto sol1a=computeGS(model1.Ham(rot, true));


    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    auto cc=Fermionic::cc_matrix(sol1a.psi, sol1a.sites);
    cc.diag().print("ni");
    rot = rot*Fermionic::rotNO(cc);
    auto sol1b=computeGS(model1.Ham(rot));

    auto psi1=sol1b.psi;
    psi1.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<psi1.length(); i++)
        cout<<itensor::leftLinkIndex(psi1,i+1).dim()<<" ";
    cout << "\n";
    return 0;

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    auto model2=IRLM {.L=len, .t=0.5, .V=0.15, .U=-0.5};
    auto psi=sol1b.psi;

    cout<<"time M m energy\n" << setprecision(12);
    auto sys2=model2.Ham(rot);
    for(auto i=0; i<100; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        auto sys2=model2.Ham(rot);
        it_tdvp sol {sys2, psi};
        sol.dt={0,0.1};
        sol.bond_dim=256;
        cout<<"0 "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        for(auto k=0; k<1; k++) {
            sol.iterate();
            cout<<(i*1+k+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<endl;
        }
        psi=sol.psi;
        psi.orthogonalize({"Cutoff",1e-9});
        cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
        cc.diag().print("ni");
        auto rot1=Fermionic::rotNO2(cc);
        psi=rotateState3(sol.psi, rot1).psi;
        psi.orthogonalize({"Cutoff",1e-9});
        rot = rot*rot1;
        for(auto i=0; i<sol.psi.length(); i++)
            cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
        cout<<endl;
    }

//    psi.replaceSiteInds(sys2.sites.inds());
//    cc=Fermionic::cc_matrix(psi, sys2.sites);
//    cc.diag().print("ni");
//    auto rot1=Fermionic::rotNO2(cc);
//    State st=rotateState(psi,rot1);
//    cc=Fermionic::cc_matrix(st.psi, st.sites);

//    auto sys=model2.Ham(rot*rot1);
//    psi=st.psi;
//    psi.replaceSiteInds(sys.sites.inds());
//    cout<<"energy="<<std::real(itensor::innerC(psi, sys.ham, psi))<<endl;

    return 0;
}
