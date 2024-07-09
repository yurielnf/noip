#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include<itensor/all.h>
#include <nlohmann/json.hpp>

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
template<class T>
arma::Mat<T> MagicRotation(arma::Mat<T> const& cc, arma::Mat<T> const& kin, double tolWannier)
{
    using namespace arma;
    Mat<T> rot(size(cc), fill::eye);
    uint nInactive=0;
    {// inactive evec of cc
       Mat<T> evec;
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
        Mat<T> X=rot1.t() * kin * rot1;
        Mat<T> evec;
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
    using namespace nlohmann;
    using matriz=arma::cx_mat;

    json j;
    {
        std::ifstream in("param.json");
        if (!in) throw invalid_argument("Please provide the param.json file");
        j=json::parse(in);
    }


    cout<<"irlm_nat_orb [len=20] [hamRestricted=1] [dt=0.1] [circuit_dt=0.1]"<<endl;
    int len=j.at("irlm").at("L"), nExclude=2;
    double U=j.at("irlm").at("U");
    auto model1=IRLM_ip {IRLM {.L=len, .t=0.5, .V=0.0, .U=U, .ed=-10, .connected=false}};
    auto model2=IRLM_ip {IRLM {.L=len, .t=0.5, .V=0.1, .U=U, .ed=0.0}};

    cout<<"\n-------------------------- solve the gs2 ----------------\n" << setprecision(15);

    matriz rot = model2.irlm.rotStar() * cx_double(1,0);
    auto sol2a=computeGS(model2.Ham(rot));

    //arma::mat xOp=arma::diagmat(arma::regspace(0,len-1));
    matriz cc=Fermionic::cc_matrix(sol2a.psi, sol2a.hamsys.sites) * cx_double(1,0);  ///<------------- real!
    //cc.save("cc_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);
    //rot.save("orb_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);

    arma::mat K;
    {
        arma::mat Umat;
        std::tie(K,Umat)=model2.irlm.matrices();
    }


    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    double tolGs2=j.at("circuit").at("tolGs2");
    {        
        matriz kin=rot.t()*K*rot;
        matriz rot1p=MagicRotation<matriz::value_type>(cc.submat(nExclude,nExclude,len-1,len-1),
                                           kin.submat(nExclude,nExclude,len-1,len-1),
                                           tolGs2);
        matriz rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(nExclude,nExclude,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }

    auto sys2b=model2.Ham(rot);
    auto sol2b=computeGS(sys2b);

    cc=Fermionic::cc_matrix(sol2b.psi, sol2b.hamsys.sites) * cx_double(1,0);
    cc.save("cc_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    int circuit_nImp=j.at("circuit").at("nImp");
    int circuit_nSite=j.at("circuit").at("nSite");
    double circuit_dt=j.at("circuit").at("dt");
    if (circuit_nImp==-1)
    {
        vec ni=arma::real(cc.diag());
        circuit_nImp=arma::find(ni>tolGs2 && ni<1-tolGs2).eval().size();  // number of active orbitals in the gs of model2
        if (circuit_nImp>12) circuit_nImp=12;
        matriz cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::vec eval=arma::eig_sym(cc1);
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-15));
        tolGs2=sqrt(activity(iev.at(circuit_nImp-nExclude-1))*activity(iev.at(circuit_nImp-nExclude))); // in the middle (logscale)
    }
    cout<<"\nNumber of active orbitals of the future gs, tol: "<<circuit_nImp<<", "<<tolGs2<<endl;

    cout<<"\n-------------------------- find the gs1 in NO2 ----------------\n";
    {        
        matriz kin=rot.t()*K*rot;
        matriz rot1p=MagicRotation<matriz::value_type>(cc.submat(circuit_nImp,circuit_nImp,len-1,len-1),
                                      kin.submat(circuit_nImp,circuit_nImp,len-1,len-1),
                                      tolGs2);
        matriz rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(circuit_nImp,circuit_nImp,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }
    auto sys1a=model1.Ham(rot);
    auto sol1a=computeGS(sys1a);
    cc=Fermionic::cc_matrix(sol1a.psi, sol1a.hamsys.sites)* cx_double(1,0);

    //cc.save("cc_L"s+to_string(len)+"_gs1.txt",arma::raw_ascii);
    //rot.save("orb_L"s+to_string(len)+"_gs1.txt",arma::raw_ascii);
    //exportPsi(sol1b.psi,"psi_t"s+to_string(0)+".txt");

    cout<<"\n-------------------------- find again the gs1 in NO1 while keeping the active NO2 ----------------\n";
    if (true) { // circuit1
        auto givens=Fermionic::NOGivensRot(cc,circuit_nImp,circuit_nSite);
        auto rot1=matrot_from_Givens(givens,cc.n_rows);
        rot = rot*rot1.st();
        cc=rot1*cc*rot1.t();
    }
    auto sys1b=model1.Ham(rot);
    auto sol1b=computeGS(sys1b);
    cc=Fermionic::cc_matrix(sol1b.psi, sol1b.hamsys.sites)* cx_double(1,0);

    cc.save("cc_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);


    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";
    double dt=j.at("tdvp").at("dt");
    double tolActivity=j.at("ip").at("tolActivity");
    auto psi=sol1b.psi;

    ofstream out("irlm_no_L"s+to_string(len)+".txt");
    out<<"time M m energy n0 cd nActive\n"<<setprecision(14);
    double n0=itensor::expectC(psi, sol1b.hamsys.sites, "N",{1}).at(0).real();
    auto cdOp=[&](itensor::Fermion const& sites) {
        itensor::AutoMPO ampo(sites);
        ampo += "Cdag",1,"C",2;
        ampo += "Cdag",2,"C",1;
        return itensor::toMPO(ampo);
    };
    double cd=itensor::innerC(psi, cdOp(sol1b.hamsys.sites), psi).real();
    vec ni=arma::real(cc.diag());
    arma::uvec active=arma::find(ni>tolActivity && ni<1-tolActivity);
    out<<"0 "<< maxLinkDim(sys1b.ham) <<" "<<maxLinkDim(sol1b.psi)<<" "<<sol1b.energy<<" "<<n0<<" "<<cd<<" "<<active.size()<<endl;
    auto model2_ip=IRLM_ip{model2};
    for(auto i=0; i*dt<=len; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        itensor::cpu_time t0;
        int nImpIp=map<string,int> {{"circuit", circuit_nImp},
                                    {"activity", active.size()},
                                    {"none", len}
                                   }.at(j.at("ip").at("type"));
        auto [sys2,givens] = model2_ip.HamIP_f(rot,nImpIp,dt);
        if (nImpIp!=len) rot = rot * model2_ip.rotIP(rot,nImpIp,dt) * matrot_from_Givens(givens,len).st();
        cout<<"Hamiltonian mpo:"<<t0.sincemark()<<endl;
        t0.mark();

        {// the circuit to extract f orbitals
            auto rot1=matrot_from_Givens(givens,len);
            auto gates=Fermionic::NOGates(sys2.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit-f:"<<t0.sincemark()<<endl;
            t0.mark();

            // matriz kin=(rot.t()*K*rot);
            // kin.submat(nImpIp,nImpIp,len-1,len-1).fill(0);
            // matriz k12=kin.submat(0,nImpIp,nImpIp-1,len-1);
            // vec s;
            // matriz U,V;
            // svd_econ(U,s,V,arma::conj(k12));
            // int nSv=arma::find(s>1e-12*s[0]).eval().size();
            // auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
            // for(auto& g:givens) g.b+=nImpIp;
            // matriz rot1=matrot_from_Givens(givens,len);
            // rot = rot*rot1.st();
            // kin.print("kin");
            // (rot1.st().t()*kin*rot1.st()).eval().clean(1e-15).print("kin after rot f");
            // return 0;
        }

        it_tdvp sol {sys2, psi};
        sol.dt={0,dt};
        sol.bond_dim=512;
        sol.rho_cutoff=1e-14;
        sol.silent=true;
        sol.epsilonM=j.at("tdvp").at("epsilonM");
        sol.nKrylov=j.at("tdvp").at("nKrylov");
        sol.enrichByFit = false; //(i%10!=0);

        sol.iterate();
        cout<<"tdvp time"<<t0.sincemark()<<endl;
        t0.mark();

        psi=sol.psi;
        cc=Fermionic::cc_matrix(psi, sol.hamsys.sites)* cx_double(1,0);
        cout<<"cc computation:"<<t0.sincemark()<<endl;
        t0.mark();

        if (std::abs(i*dt-std::round(i*dt/circuit_dt)*circuit_dt) < 0.5*dt) {        
            auto givens=Fermionic::NOGivensRot(cc,circuit_nImp,circuit_nSite);
//            auto givens=Fermionic::GivensRotForMatrix(cc,circuit_nImp,20);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            //real((rot1 * cc * rot1.t()).eval().clean(1e-10).submat(circuit_nImp,circuit_nImp,cc.n_rows-1,cc.n_cols-1)).print("rot1*cc*rot1.t()");
            auto gates=Fermionic::NOGates(sol.hamsys.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit1:"<<t0.sincemark()<<endl;
            t0.mark();
            //matriz ccr=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //real(ccr.clean(1e-7).submat(circuit_nImp,circuit_nImp,cc.n_rows-1,cc.n_cols-1)).print("cc after rot");
            rot = rot*rot1.st();
            cc=rot1*cc*rot1.t();
            psi.orthogonalize({"Cutoff",1e-9});
            vec ni=arma::real(cc.diag());
            active=arma::find(ni>tolActivity && ni<1-tolActivity);

            cout<<"active: "<<active.size()<<endl;

            for(auto i=0; i<psi.length(); i++)
                cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
            cout<<endl;
        }


        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        cd=itensor::innerC(sol.psi, cdOp(sol.hamsys.sites), sol.psi).real();
        out<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(psi)<<" "<<sol.energy<<" "<<n0<<" "<<cd<<" "<<active.size()<<endl;

        if (i>0 && i%100==0) {
            cc.save("cc_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            rot.save("orb_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            //exportPsi(psi,"psi_t"s+to_string(i)+".txt");
        }
    }

    return 0;
}
