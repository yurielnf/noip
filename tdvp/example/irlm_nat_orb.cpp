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

    itensor::Fermion sites(psi.length(), {"ConserveNf=",true});
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
    return sol_gs;
}


static arma::mat TestrotNO4(arma::mat const& orb, arma::mat const& cc, int nExclude=2, double tolWannier=1e-5)
{
    using namespace arma;
    arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
    arma::mat orb1=orb.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
    arma::mat J=orb1.t() * arma::diagmat(arma::regspace(0,cc1.n_rows-1)) * orb1;


    // Wannier after activity sorting
    arma::uvec active;
    arma::vec Xeval;
    arma::mat Xevec;

    {// group active natural orbitals
        active=arma::find(cc1.diag()>tolWannier && cc1.diag()<1-tolWannier);
        arma::mat X=J;//(active,active);
        arma::eig_sym(Xeval,Xevec,X);
    }

    // apply Wannierization

    // sort Wannier orbitals according to position
    arma::uvec Xiev(Xeval.size());
    {
        Xiev=arma::stable_sort_index(Xeval);
//            for(auto i=0u; i<Xeval.size(); i++) Xiev[i]=i;
    }

    arma::mat evec4=Xevec.cols(Xiev);

    arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
    rot.submat(nExclude,nExclude,arma::size(evec4))=evec4;

    Xeval.print("xeval");
    return rot;
}

void TestGivens()
{
    { // 2d case
        arma::vec v={0.5,1.5};
        auto g=GivensRot(0).make(v[0],v[1]);
        g.matrix().print("g");
        (g.matrix()*v).print("g*v");
    }
    { // 3d case
        vector<GivensRot> gs;
        arma::vec v={0.5,1.5,-1}, vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot(b).make(v[i],v[i+1]);
            gs.push_back(g);
            v[i+1]=g.r;
        }
        auto rot=matrot_from_Givens(gs);
        (rot*vc).print("rot*v");
    }
    { // matrix case
        vector<GivensRot> gs;
        arma::mat A(3,3, arma::fill::randu), evec;
        arma::vec eval;
        A = A*A.t();
        arma::eig_sym(eval,evec,A);
        arma::vec v=evec.col(0), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot(b).make(v[i],v[i+1]);
            gs.push_back(g);
            v[i+1]=g.r;
        }
        auto rot=matrot_from_Givens(gs);
        (rot*vc).print("rot*v");
        (rot*A*rot.t()).print("rot.t()*A*rot");
    }
}


itensor::BondGate BondGateFromAngle(itensor::Fermion const& sites, int b, double angle)
{
    using namespace itensor;
    b=b+1;  // itensor convention
    auto hterm = ( sites.op("Adag",b)*sites.op("A",b+1)
                   -sites.op("Adag",b+1)*sites.op("A",b))* (angle*Cplx_i);
    return BondGate(sites,b,b+1,BondGate::tReal,1,hterm);
}

double BondEntropy(itensor::MPS &psi, itensor::Fermion const& sites, int b, double angle)
{
    using namespace itensor;
    auto gate=BondGateFromAngle(sites,b,angle);
    psi.position(b+1); // itensor convention b+1

    auto AA = psi(gate.i1())*psi(gate.i2())*gate.gate();
    AA.noPrime();
    auto [U,S,V] = svd(AA,inds(psi(gate.i1())),{"Cutoff=",1E-8, "MaxDim", rightLinkIndex(psi,gate.i1()).size()});
    auto u = commonIndex(U,S);

    //Apply von Neumann formula
    //to the squares of the singular values
    Real SvN = 0.;
    for(auto n : range1(dim(u)))
    {
        auto Sn = elt(S,n,n);
//        auto p=sqr(Sn);
//        if(Sn > 1E-12) SvN += -p*log(p);
        if(Sn > 1E-12) SvN += Sn;
    }
    return SvN;
}


void exportPsi(itensor::MPS const& psi, string name="psi.txt")
{
    using namespace itensor;

    ofstream out(name);
    out<<setprecision(14);
    out<<psi.length()<<endl;
    for(auto i:range1(psi.length()-1))
        out<<" "<<rightLinkIndex(psi,i).size();
    out<<endl<<endl;
    for(auto i:range1(psi.length())) {
        ITensor M=psi.A(i);
        auto li=leftLinkIndex(psi,i);
        auto si=siteIndex(psi,i);
        auto ri=rightLinkIndex(psi,i);

        if (i==1)
            for(auto b : range1(si))
                for(auto c : range1(ri))
                    print(out," ",eltC(M,si=b,ri=c));
        else if (i==psi.length())
            for(auto a : range1(li))
                for(auto b : range1(si))
                    print(out," ",eltC(M,li=a,si=b));
        else
            for(auto a : range1(li))
                for(auto b : range1(si))
                    for(auto c : range1(ri))
                        print(out," ",eltC(M,li=a,si=b,ri=c));
        out<<endl<<endl;
    }
}

arma::mat optimizeEntropyPair(itensor::MPS &psi, itensor::Fermion const& sites, int b, int nangle=20)
{
    double dx=M_PI/nangle;
    vector<double> s(nangle);
    for(auto i=0; i<nangle; i++) s[i]=BondEntropy(psi,sites,b,-M_PI/2+dx*i);
    auto pos=std::min_element(s.begin(),s.end())-s.begin();

    if (s[nangle/2]-s[pos]<1e-5) return {};
    auto angle=-M_PI/2+dx*pos;
    cout<<"\noptimization at bond "<<b<<" angle="<<angle<<" s[0]="<<s[nangle/2]<<" s[angle]=="<<s[pos]<<endl;
    auto gate=BondGateFromAngle(sites,b,angle);
    gateTEvol(vector{gate},1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
    //rotate the H    <-------!!!
    arma::mat rot1(psi.length(), psi.length(), arma::fill::eye);
    rot1.submat(b,b,b+1,b+1)={{cos(angle),sin(angle)},{-sin(angle),cos(angle)}};
    return rot1;
}


/// ./irlm_nat_orb <len>
int main(int argc, char **argv)
{
//    TestGivens();
    int len=20, nExclude=2;
    double tolWannier=1e-9;
    if (argc==2) len=atoi(argv[1]);
    cout<<"\n-------------------------- solve the gs of system ----------------\n" << setprecision(12);

    auto model1=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=-10};
    auto model2=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=0.0};
    auto rot=model2.rotStar();
    //eig_unitary(rot);
    //return 0;
    auto sol1a=computeGS(model2.Ham(rot, true));

    cout<<"\n-------------------------- rotate the H to natural orbitals: find the gs again ----------------\n";

    auto cc=Fermionic::cc_matrix(sol1a.psi, sol1a.hamsys.sites);
    cc.diag().print("ni");
    rot = rot*Fermionic::rotNO3(cc,nExclude,tolWannier);
    auto sys1b=model1.Ham(rot, nExclude==2);
    auto sol1b=computeGS(sys1b);
    //cc=Fermionic::cc_matrix(sol1b.psi, sol1b.hamsys.sites);
    //cc.diag().raw_print("ni");
    arma::vec eval=arma::eig_sym(cc);
    int nExcludeGs=arma::find(eval>tolWannier && eval<1-tolWannier).eval().size();  // number of active orbitals in the gs of model2
    cout<<"\nNumber of active orbitals of the future gs: "<<nExcludeGs<<endl;
    nExcludeGs=2;

    auto psi1=sol1b.psi;
    psi1.orthogonalize({"Cutoff",1e-9});
    for(auto i=0; i<psi1.length(); i++)
        cout<<itensor::leftLinkIndex(psi1,i+1).dim()<<" ";
    cout << "\n";

    // for Maxime
    cc.save("cc_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    exportPsi(sol1b.psi,"psi_t"s+to_string(0)+".txt");

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";

    ofstream out("irlm_no_L"s+to_string(len)+".txt");
    out<<"time M m energy n0\n"<<setprecision(14);
    double n0=itensor::expectC(sol1b.psi, sol1b.hamsys.sites, "N",{1}).at(0).real();
    out<<"0 "<< maxLinkDim(sys1b.ham) <<" "<<maxLinkDim(sol1b.psi)<<" "<<sol1b.energy<<" "<<n0<<endl;
    auto psi=sol1b.psi;
    arma::uvec inactive;
    for(auto i=0; i<len*10/2; i++) {
        cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        itensor::cpu_time t0;
        auto sys2=model2.Ham(rot, nExclude==2, inactive);
        cout<<"Hamiltonian mpo:"<<t0.sincemark()<<endl;
        t0.mark();
        it_tdvp sol {sys2, psi};
        sol.dt={0,0.1};
        sol.bond_dim=512;
        sol.rho_cutoff=1e-14;
        sol.silent=false;
        sol.epsilonM=(i%1==0) ? 1e-4 : 0;
        sol.enrichByFit = (i%10!=0);


        sol.iterate();
        cout<<"tdvp time"<<t0.sincemark()<<endl;
        t0.mark();

        psi=sol.psi;
        //psi.orthogonalize({"Cutoff",1e-9});
        cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
        cc.diag().print("ni");
        inactive=arma::find(cc.diag()<1e-5 || cc.diag()>1-1e-5);
        cout<<"active: "<<len-inactive.size()<<" cc computation:"<<t0.sincemark()<<endl;
        t0.mark();
        //double n0=arma::cdot(rot.row(0), cc*rot.row(0).st());
        if (true || i%10==9) {
            //auto rot1=Fermionic::rotNO3(cc,nExclude);
//            psi=rotateState3(psi, rot1, nExclude).psi;
            auto gs=Fermionic::NOGivensRot(cc,nExcludeGs,8);
            auto rot1=matrot_from_Givens(gs);            
            //(rot1 * cc * rot1.t()).print("rot1*cc*rot1.t()");
            auto gates=Fermionic::NOGates(sol.hamsys.sites,gs);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit:"<<t0.sincemark()<<endl;
            t0.mark();
            //cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //cc.print("cc after rot");
            rot = rot*rot1.t();
        }

        if (false && i%10==0) {// try Wannier of acitve orbitals:
            TestrotNO4(rot,cc,nExclude);
            auto rot1=Fermionic::rotNO4(rot,cc,nExclude);
            auto psi2=rotateState3(psi, rot1, nExclude).psi;
            psi2.orthogonalize({"Cutoff",1e-9});
            cout<<"Wannier of active orbitals:\n";
            for(auto i=0; i<psi2.length(); i++)
                cout<<itensor::leftLinkIndex(psi2,i+1).dim()<<" ";
            cout<<endl;
        }

        if (false && i%10==9) {// optimize the 2-site entropy
            bool found=true;
            for(auto i=0;i<20 && found; i++) {
                found=false;
                for(int b=len-inactive.size()-1; b>=nExclude; b--) {
                    auto rot1=optimizeEntropyPair(psi,sol.hamsys.sites,b);
                    if (!rot1.empty()) { rot=rot*rot1.t(); found=true; }
                }
                if (!found) break;
                found=false;
                for(auto b=nExclude; b<len-inactive.size(); b++) {
                    auto rot1=optimizeEntropyPair(psi,sol.hamsys.sites,b);
                    if (!rot1.empty()) { rot=rot*rot1.t(); found=true; }
                }
            }
        }

        psi.orthogonalize({"Cutoff",1e-9});
        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        out<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham) <<" "<<maxLinkDim(psi)<<" "<<sol.energy<<" "<<n0<<endl;

        if (true && i%100==0) {
            cc.save("cc_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            rot.save("orb_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            exportPsi(psi,"psi_t"s+to_string(i)+".txt");
        }

        for(auto i=0; i<psi.length(); i++)
            cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
        cout<<endl;
    }

    return 0;
}
