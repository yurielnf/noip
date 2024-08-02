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
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-3;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<30; i++) {
        if (i==10) { sol_gs.noise=1e-7; sol_gs.bond_dim=256; }
        else if (i==26) { sol_gs.noise=1e-9; sol_gs.bond_dim=512; }
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


    bool verbose=j.at("verbose");
    int len=j.at("irlm").at("L"), nExclude=2;
    double U=j.at("irlm").at("U");
    itensor::Fermion sites(len, {"ConserveNf",true});
    auto model1=IRLM_ip {sites, IRLM {.L=len, .t=0.5, .V=0.0, .U=U, .ed=-10, .connected=false}};
    auto model2=IRLM_ip {sites, IRLM {.L=len, .t=0.5, .V=0.1, .U=U, .ed=0.0}};

    cout<<"\n-------------------------- solve the gs2 ----------------\n" << setprecision(15);

    matriz rot = model2.irlm.rotStar() * cx_double(1,0);
    auto sol2a=computeGS(model2.Ham(rot));

    //arma::mat xOp=arma::diagmat(arma::regspace(0,len-1));
    matriz cc=Fermionic::cc_matrix(sol2a.psi, sol2a.hamsys.sites) * cx_double(1,0);
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
    // double circuit_dt=j.at("circuit").at("dt");
    double circuit_tol=j.at("circuit").at("tol");
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
    cc=Fermionic::cc_matrix(sol1a.psi, sites)* cx_double(1,0);

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


    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n"; cout.flush();
    double dt=j.at("tdvp").at("dt");
    double t0=j.at("green").at("t0");
    bool greater=j.at("green").at("greater");
    double tolActivity=j.at("ip").at("tolActivity");
    auto psi=sol1b.psi;



    auto evolve=[&](MPS &psi, matriz &rot, int &nActive, matriz& cc,double dt) {
        itensor::cpu_time t0;
        int nImpIp=map<string,int> {{"circuit", circuit_nImp},
                                    {"activity", std::max(circuit_nImp,nActive)},
                                    {"none", len}
                                   }.at(j.at("ip").at("type"));
        // auto [sys2,givens,Kip] = model2_ip.HamIP_f(rot,nImpIp,dt);
        auto [sys2,givens,Kip] = model2.HamIPS(rot, nImpIp, dt, j.at("extract_f"));
        psi.replaceSiteInds(sys2.sites.inds());
        // if (nImpIp!=len) rot = rot * model2_ip.rotIP(rot,nImpIp,dt) * matrot_from_Givens(givens,len).st();
        if (nImpIp!=len) { rot = rot * model2.rotIPS(rot,nImpIp,dt,cc) * matrot_from_Givens(givens,len).st(); }
        if (verbose) cout<<"Hamiltonian mpo:"<<t0.sincemark()<<endl;
        t0.mark();

        if (j.at("extract_f")) {// the circuit to extract f orbitals
            auto rot1=matrot_from_Givens(givens,len);
            auto gates=Fermionic::NOGates(sys2.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",circuit_tol,"Quiet",true, "DoNormalize",true,"ShowPercent",false});
            if (verbose) cout<<"circuit-f:"<<t0.sincemark()<<endl;
            t0.mark();
        }

        if (j.at("use_tdvp")) {// tdvp
            it_tdvp sol {sys2, psi};
            sol.dt={0,dt};
            sol.bond_dim=512;
            sol.rho_cutoff=1e-14;
            sol.silent=true;
            sol.epsilonM=j.at("tdvp").at("epsilonM");
            sol.nKrylov=j.at("tdvp").at("nKrylov");
            sol.enrichByFit = false; //(i%10!=0);

            sol.iterate();
            psi=sol.psi;
        }
        else {
            auto gates=model2.TrotterGatesExp(Kip,3,dt);
            gateTEvol(gates,1,1,psi,{"Cutoff=",circuit_tol,"Quiet=",true, "DoNormalize",true,"ShowPercent",false});
        }

        if (verbose) cout<<"tdvp time"<<t0.sincemark()<<endl;
        t0.mark();


        cc=Fermionic::cc_matrix(psi, sys2.sites)* cx_double(1,0);
        if (verbose) cout<<"cc computation:"<<t0.sincemark()<<endl;
        t0.mark();

        if (true) {
            auto givens=Fermionic::NOGivensRot(cc,circuit_nImp,circuit_nSite);
//            auto givens=Fermionic::GivensRotForMatrix(cc,circuit_nImp,20);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            auto gates=Fermionic::NOGates(sys2.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",circuit_tol,"Quiet",true, "DoNormalize",true,"ShowPercent",false});
            if (verbose) cout<<"circuit1:"<<t0.sincemark()<<endl;
            t0.mark();
            rot = rot*rot1.st();
            cc=rot1*cc*rot1.t();
            vec ni=arma::real(cc.diag());
            nActive=arma::find(ni>tolActivity && ni<1-tolActivity).eval().size();

            if (verbose) {
                cout<<"active: "<<nActive<<endl;
                for(auto i=0; i<psi.length(); i++)
                    cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
                cout<<endl;
            }
        }
    };

    ofstream out("irlm_no_green_L"s+to_string(len)+".txt");
    out<<"time mBra mKet green nActiveBra nActiveKet\n"<<setprecision(14);
    vec ni=arma::real(cc.diag());
    int nActive=arma::find(ni>tolActivity && ni<1-tolActivity).eval().size();

    for(auto i=0; i*dt<t0; i++) {
        evolve(psi,rot,nActive,cc,dt);
    }

    auto compute_green=[&](bool is_forward)
    {
        MPS bra=psi, ket=psi;
        matriz rotB=rot, rotK=rot;
        int nActiveB=nActive, nActiveK=nActive;
        matriz ccB=cc, ccK=cc;

        {// apply the excitation
            ket.position(1);
            auto newA = op(sites, (greater ? "Adag": "A"), 1)* ket(1);
            newA.noPrime();
            ket.ref(1)=newA;
        }

        double mydt= is_forward ? dt : -dt;
        for(auto i=0; ; i++) {
            if (is_forward && (t0+i*dt>=len)) break;
            if (!is_forward && (t0-i*dt<=0.0)) break;
            if (verbose) cout<<"-------------------------- iteration "<<i+1<<" --------\n";
            evolve(bra,rotB,nActiveB,ccB,mydt);
            evolve(ket,rotK,nActiveK,ccK,mydt);

            MPS ket2=ket;
            {// apply the excitation
                ket2.position(1);
                auto newA = op(sites, (greater ? "A": "Adag"), 1)* ket2(1);
                newA.noPrime();
                ket2.ref(1)=newA;
            }

            complex<double> g;
            {// exp(Q) * ket2
                auto givens=GivensRotForRot_left((rotB.t()*rotK).t().st().eval());
                auto gates=Fermionic::NOGates(sites,givens);
                gateTEvol(gates,1,1,ket2,{"Cutoff",1e-4,"Quiet",true, "DoNormalize",true,"ShowPercent",false});
                g=itensor::innerC(bra,ket2);
            }
            out<<t0+(i+1)*mydt <<" "<<maxLinkDim(bra)<<" "<<maxLinkDim(ket)<<" "<<g.real()<<" "<<g.imag()<<" "<<nActiveB<<" "<<nActiveK<<endl;
        }
    };

    compute_green(true);
    compute_green(false);

    return 0;
}
