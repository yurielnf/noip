#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

#include<itensor/all.h>

using namespace std;

struct State {
    itensor::MPS psi;
    itensor::Fermion sites;
};

auto computeGS(HamSys const& sys)
{
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=512;
    sol_gs.noise=1e-5;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<30; i++) {
        if (i==10) sol_gs.noise=1e-7;
        else if (i==26) sol_gs.noise=1e-9;
        else if (i==28) sol_gs.noise=0;
        if (false && i==11) {
            // Global subspace expansion
//            std::vector<double> epsilonK(3,1E-8);
            std::vector<int> maxDimK(3,0.5*itensor::maxLinkDim(sol_gs.psi));

            itensor::addBasis(sol_gs.psi,sol_gs.hamsys.ham,maxDimK,
                     {"Cutoff", 1e-9,
                      "Method", "DensityMatrix",
                      "KrylovOrd",3,
                      "DoNormalize", true,
                      "Quiet",true,
                      "Silent",true});
        }
        sol_gs.iterate();
        //sol_gs.psi.orthogonalize({"Cutoff",1e-9});
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return sol_gs;
}


/// return a rotation rot such that rot(:,B).t()*kin*rot(:,B) is diagonal where B are the inactive eigenvectors of cc (with eval=0 or 1).
arma::cx_mat MagicRotation(arma::cx_mat const& cc, arma::cx_mat const& kin, double tolWannier)
{
    using namespace arma;
    cx_mat rot(size(cc), fill::eye);
    uint nInactive=0;
    {// inactive evec of cc
       cx_mat evec;
       vec eval;
       eig_sym(eval,evec,cc);
       { //sort by activity
           arma::vec activity(eval.size());
           for(auto i=0u; i<eval.size(); i++)
               activity[i]=-std::min(eval[i], -eval[i]+1);
           arma::uvec iev=arma::stable_sort_index(activity.clean(1e-15)); //activity sorting
           eval=eval(iev);
           evec=evec.cols(iev);
       }
       uvec active=arma::find(eval>tolWannier && eval<1-tolWannier);
       uvec inactive=arma::find(eval<=tolWannier || eval>=1-tolWannier);
       rot.head_cols(active.size())=evec.cols(active);                      // TODO: Wannierize the active sector (?)
       rot.tail_cols(inactive.size())=evec.cols(inactive);
       nInactive=inactive.size();
    }
    {// diagonalize kin in the subspace of inactive orbitals
        auto rot1=rot.tail_cols(nInactive);  // this is a view!
        cx_mat X=rot1.t() * kin * rot1;
        cx_mat evec;
        vec eval;
        eig_sym(eval,evec,X);
        { //sort by position
            arma::mat xOp=arma::diagmat(arma::regspace(0,cc.n_rows-1));
            vec x=arma::real((evec.t()*rot1.t()*xOp*rot1*evec).eval().diag());
            uvec iev=stable_sort_index(x);
            evec=evec.cols(iev).eval();
        }
        rot1=rot1*evec;
    }

    // cout<<"n inactive = "<<nInactive<<endl;
    // cout<<"rot.t * rot-1 ="<<norm(rot.t()*rot-mat(size(rot),fill::eye))<<endl;
    // (rot.t()*cc*rot).eval().clean(1e-15).print("cc rotated");
    // (rot.t()*kin*rot).eval().clean(1e-15).print("kin rotated");

    // fix the sign
    for(auto j=0u; j<rot.n_cols; j++) {
        auto i=arma::index_max(arma::abs(rot.col(j)));
        rot.col(j) /= arma::sign(rot(i,j));
    }

    cout<<"norm(1-rot)="<<norm(rot-arma::mat(size(rot),fill::eye))<<endl;

    return rot;
}


/// ./irlm_nat_orb [len=20] [hamRestricted=1] [dt=0.1] [circuit_dt=0.1]
int main(int argc, char **argv)
{
    using namespace itensor;
    using namespace arma;    

    cout<<"irlm_nat_orb [len=20] [hamRestricted=1] [dt=0.1] [circuit_dt=0.1]"<<endl;
//    TestGivens();
    int len=20, nExclude=2;
    bool hamRestricted=true;
    double dt=0.1, circuit_dt=0.1;
    if (argc>=2) len=atoi(argv[1]);
    if (argc>=3) hamRestricted=atoi(argv[2]);
    if (argc>=4) dt=atof(argv[3]);
    if (argc>=5) circuit_dt=atof(argv[4]);
    auto model1=IRLM_ip {IRLM {.L=len, .t=0.5, .V=0.0, .U=0.25, .ed=-10, .connected=false}};
    auto model2=IRLM_ip {IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=0.0}};

    cout<<"\n-------------------------- solve the gs2 ----------------\n" << setprecision(15);

    arma::cx_mat rot = model2.irlm.rotStar() * cx_double(1,0);
    auto sol2a=computeGS(model2.Ham(rot));

    //arma::mat xOp=arma::diagmat(arma::regspace(0,len-1));
    auto cc=Fermionic::cc_matrix(sol2a.psi, sol2a.hamsys.sites);
    //cc.save("cc_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);
    //rot.save("orb_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);

    arma::mat K;
    {
        arma::mat Umat;
        std::tie(K,Umat)=model2.irlm.matrices();
    }


    // arma::mat rm=MagicRotation(cc.submat(nExclude,nExclude,len-1,len-1),
    //                            rot.tail_cols(len-nExclude).t()*K*rot.tail_cols(len-nExclude));
    // auto givens=Fermionic::GivensRotForRot(rm);
    // for(auto &g : givens) g.b+=nExclude;
    // return 0;

    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    double tolWannier=1e-5;
    int nExcludeGs=nExclude;
    //auto rot1=Fermionic::rotNO3(cc,xOp,nExclude,tolWannier);
    //(rot1.t()*cc*rot1).eval().save("cc_L"s+to_string(len)+"_gs2_star_noW.txt",arma::raw_ascii);
    {        
        arma::cx_mat kin=rot.t()*K*rot;
        arma::cx_mat rot1p=MagicRotation(cc.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      kin.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      tolWannier);
        arma::cx_mat rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(nExcludeGs,nExcludeGs,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }

    auto sys2b=model2.Ham(rot);
    auto sol2b=computeGS(sys2b);

    cc=Fermionic::cc_matrix(sol2b.psi, sol2b.hamsys.sites);
    cc.save("cc_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    {
        auto ni=arma::real(cc.diag());
        nExcludeGs=arma::find(ni>tolWannier && ni<1-tolWannier).eval().size();  // number of active orbitals in the gs of model2
        if (nExcludeGs>12) nExcludeGs=12;
        arma::cx_mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::vec eval=arma::eig_sym(cc1);
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-15));
        tolWannier=sqrt(activity(iev.at(nExcludeGs-nExclude-1))*activity(iev.at(nExcludeGs-nExclude))); // in the middle (logscale)
    }
    cout<<"\nNumber of active orbitals of the future gs, tol: "<<nExcludeGs<<", "<<tolWannier<<endl;

    cout<<"\n-------------------------- find the gs1 in NO2 ----------------\n";
    {        
        arma::cx_mat kin=rot.t()*K*rot;
        arma::cx_mat rot1p=MagicRotation(cc.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      kin.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      tolWannier);
        arma::cx_mat rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(nExcludeGs,nExcludeGs,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }
    auto sys1a=model1.Ham(rot);
    auto sol1a=computeGS(sys1a);
    cc=Fermionic::cc_matrix(sol1a.psi, sol1a.hamsys.sites);

    //cc.save("cc_L"s+to_string(len)+"_gs1.txt",arma::raw_ascii);
    //rot.save("orb_L"s+to_string(len)+"_gs1.txt",arma::raw_ascii);
    //exportPsi(sol1b.psi,"psi_t"s+to_string(0)+".txt");

    cout<<"\n-------------------------- find again the gs1 in NO1 while keeping the active NO2 ----------------\n";
    if (true) { // circuit1
        auto givens=Fermionic::NOGivensRot(cc,nExcludeGs,40);
        auto rot1=matrot_from_Givens(givens,cc.n_rows);
        rot = rot*rot1.t();
        cc=rot1*cc*rot1.t();
    }
    auto sys1b=model1.Ham(rot);
    auto sol1b=computeGS(sys1b);
    cc=Fermionic::cc_matrix(sol1b.psi, sol1b.hamsys.sites);

    cc.save("cc_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);


    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";
    tolWannier=1e-9;
    auto psi=sol1b.psi;

    ofstream out("irlm_no_L"s+to_string(len)+".txt");
    out<<"time M m energy n0\n"<<setprecision(14);
    double n0=itensor::expectC(psi, sol1b.hamsys.sites, "N",{1}).at(0).real();
    auto cdOp=[&](itensor::Fermion const& sites) {
        itensor::AutoMPO ampo(sites);
        ampo += "Cdag",1,"C",2;
        ampo += "Cdag",2,"C",1;
        return itensor::toMPO(ampo);
    };
    double cd=itensor::innerC(psi, cdOp(sol1b.hamsys.sites), psi).real();
    out<<"0 "<< maxLinkDim(sys1b.ham) <<" "<<maxLinkDim(sol1b.psi)<<" "<<sol1b.energy<<" "<<n0<<" "<<cd<<endl;
    auto ni=arma::real(cc.diag());
    arma::uvec inactive=arma::find(ni<=tolWannier || ni>=1-tolWannier);
    auto model2_ip=IRLM_ip{model2};
    arma::cx_mat rotg=rot*arma::cx_double(1.0);
    for(auto i=0; i*dt<=len; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        itensor::cpu_time t0;
        auto sys2 = hamRestricted ?
                    model2_ip.HamIP(rot,len-inactive.size(),dt) :
                    model2_ip.Ham(rot) ;
        cout<<"Hamiltonian mpo:"<<t0.sincemark()<<endl;
        t0.mark();
        it_tdvp sol {sys2, psi};
        sol.dt={0,dt};
        sol.bond_dim=512;
        sol.rho_cutoff=1e-14;
        sol.silent=true;
        sol.epsilonM=(i%1==0) ? 1e-7 : 0;
        sol.enrichByFit = false; //(i%10!=0);

        sol.iterate();
        cout<<"tdvp time"<<t0.sincemark()<<endl;
        t0.mark();

        psi=sol.psi;
        cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);        
        cout<<"cc computation:"<<t0.sincemark()<<endl;
        t0.mark();
        if (std::abs(i*dt-std::round(i*dt/circuit_dt)*circuit_dt) < 0.5*dt) {        
            auto givens=Fermionic::NOGivensRot(cc,nExcludeGs,40);
//            auto givens=Fermionic::GivensRotForMatrix(cc,nExcludeGs,20);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            //(rot1.t() * cc * rot1).print("rot1.t()*cc*rot1");
            auto gates=Fermionic::NOGates(sol.hamsys.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit1:"<<t0.sincemark()<<endl;
            t0.mark();
            //cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //cc.print("cc after rot");
            rotg = rotg*rot1.t();
            cc=rot1*cc*rot1.t();
            psi.orthogonalize({"Cutoff",1e-9});
            auto ni=arma::real(cc.diag());
            inactive=arma::find(ni<=tolWannier || ni>=1-tolWannier);

            cout<<"active: "<<len-inactive.size()<<endl;

            for(auto i=0; i<psi.length(); i++)
                cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
            cout<<endl;
        }


        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        cd=itensor::innerC(sol.psi, cdOp(sol.hamsys.sites), sol.psi).real();
        out<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(psi)<<" "<<sol.energy<<" "<<n0<<" "<<cd<<endl;

        if (i>0 && i%100==0) {
            cc.save("cc_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            rot.save("orb_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            //exportPsi(psi,"psi_t"s+to_string(i)+".txt");
        }
    }

    return 0;
}
