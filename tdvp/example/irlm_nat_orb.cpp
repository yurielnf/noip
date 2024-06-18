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
//    arma::cx_mat(evec*diagmat(arma::log(eval)*im)*evec.t()).clean(tolWannier).print("rotHam");

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

/// return a rotation rot such that rot(:,B).t()*kin*rot(:,B) is diagonal where B are the inactive eigenvectors of cc (with eval=0 or 1).
arma::mat MagicRotation(arma::mat const& cc, arma::mat const& kin, double tolWannier)
{
    using namespace arma;
    mat rot(size(cc), fill::eye);
    uint nInactive=0;
    {// inactive evec of cc
       mat evec;
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
        mat X=rot1.t() * kin * rot1;
        mat evec;
        vec eval;
        eig_sym(eval,evec,X);
        { //sort by position
            arma::mat xOp=arma::diagmat(arma::regspace(0,cc.n_rows-1));
            vec x=(evec.t()*rot1.t()*xOp*rot1*evec).eval().diag();
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


/// return a rotation to diagonize kin. The evec are sorted according to position
arma::mat InactiveStarRotation(arma::mat const& kin)
{
    using namespace arma;
    mat evec;
    vec eval;
    eig_sym(eval,evec,kin);

    // sort by position
    arma::mat xOp=arma::diagmat(arma::regspace(0,kin.n_rows-1));
    vec x=(evec.t()*xOp*evec).eval().diag();
    uvec iev=stable_sort_index(x);
    mat rot=evec.cols(iev);

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
    cout<<"irlm_nat_orb [len=20] [hamRestricted=1] [dt=0.1] [circuit_dt=0.1]"<<endl;
//    TestGivens();
    int len=20, nExclude=2;
    bool hamRestricted=true;
    double dt=0.1, circuit_dt=0.1;
    if (argc>=2) len=atoi(argv[1]);
    if (argc>=3) hamRestricted=atoi(argv[2]);
    if (argc>=4) dt=atof(argv[3]);
    if (argc>=5) circuit_dt=atof(argv[4]);
    auto model1=IRLM {.L=len, .t=0.5, .V=0.0, .U=0.25, .ed=-10, .connected=false};
    auto model2=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.25, .ed=0.0};

    cout<<"\n-------------------------- solve the gs2 ----------------\n" << setprecision(15);

    auto rot=model2.rotStar();
    auto sol2a=computeGS(model2.Ham(rot, true));

    //arma::mat xOp=arma::diagmat(arma::regspace(0,len-1));
    auto cc=Fermionic::cc_matrix(sol2a.psi, sol2a.hamsys.sites);
    //cc.save("cc_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);
    //rot.save("orb_L"s+to_string(len)+"_gs2_star.txt",arma::raw_ascii);

    arma::mat K;
    {
        arma::mat Umat;
        std::tie(K,Umat)=model2.matrices();
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
        arma::mat kin=rot.t()*K*rot;
        arma::mat rot1p=MagicRotation(cc.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      kin.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      tolWannier);
        arma::mat rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(nExcludeGs,nExcludeGs,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }

    auto sys2b=model2.Ham(rot, nExclude==2);
    auto sol2b=computeGS(sys2b);

    cc=Fermionic::cc_matrix(sol2b.psi, sol2b.hamsys.sites);
    cc.save("cc_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    rot.save("orb_L"s+to_string(len)+"_gs2.txt",arma::raw_ascii);
    nExcludeGs=arma::find(cc.diag()>tolWannier && cc.diag()<1-tolWannier).eval().size();  // number of active orbitals in the gs of model2
    {
        if (nExcludeGs>12) nExcludeGs=12;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
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
        arma::mat kin=rot.t()*K*rot;
        arma::mat rot1p=MagicRotation(cc.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      kin.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                      tolWannier);
        arma::mat rot1(arma::size(rot),arma::fill::eye);
        rot1.submat(nExcludeGs,nExcludeGs,len-1,len-1)=rot1p;
        rot = rot*rot1;
    }
    auto sys1a=model1.Ham(rot, nExclude==2);
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
    auto sys1b=model1.Ham(rot, nExclude==2);
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
    arma::uvec inactive=arma::find(cc.diag()<=tolWannier || cc.diag()>=1-tolWannier);
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
        //psi.orthogonalize({"Cutoff",1e-9});
        cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);        
        cout<<"cc computation:"<<t0.sincemark()<<endl;
        t0.mark();
        //double n0=arma::cdot(rot.row(0), cc*rot.row(0).st());
        if (std::abs(i*dt-std::round(i*dt/circuit_dt)*circuit_dt) < 0.5*dt) {
            //auto rot1=Fermionic::rotNO3(cc,nExclude);
//            psi=rotateState3(psi, rot1, nExclude).psi;
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
            inactive=arma::find(cc.diag()<=tolWannier || cc.diag()>=1-tolWannier);

            cout<<"active: "<<len-inactive.size()<<endl;

            for(auto i=0; i<psi.length(); i++)
                cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
            cout<<endl;
        }

        if (false) { // another circuit to diagonalize kin in the inactive sector
            cc.diag().print("ni");
            arma::mat kin=rot.t()*K*rot;
            arma::uvec active=arma::find(cc.diag()>tolWannier && cc.diag()<1-tolWannier);
            cout<<"active after circuit1: "<<active.size()<<endl;
            auto givens=Fermionic::GivensRotForMatrix(kin,active.size(),16);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            //(rot1 * kin * rot1.t()).print("rot1.t()*kin*rot1");
            //std::reverse(givens.begin(),givens.end());
            //for(auto& g:givens) g.transposeInPlace();
            auto gates=Fermionic::NOGates(sol.hamsys.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit2:"<<t0.sincemark()<<endl;
            t0.mark();
            rot = rot*rot1.t();
            cc=rot1*cc*rot1.t();
        }

        if (false) {// the magic circuit!
            arma::mat kin=rot.t()*K*rot;
            arma::mat magicr=MagicRotation(cc.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                           kin.submat(nExcludeGs,nExcludeGs,len-1,len-1),
                                           tolWannier);
            auto givens=Fermionic::GivensRotForRot(magicr);
            for(auto &g : givens) g.b+=nExcludeGs;
//            auto givens=Fermionic::GivensRotForMatrix(cc,nExcludeGs,20);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            //(rot1.t() * cc * rot1).print("rot1.t()*cc*rot1");
            auto gates=Fermionic::NOGates(sol.hamsys.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit:"<<t0.sincemark()<<endl;
            t0.mark();
            //cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //cc.print("cc after rot");
            rot = rot*rot1.t();
            cc=rot1*cc*rot1.t();
        }

        if (false && i%10==9) {// diagonalize kin inactive
            int nActive=arma::find(cc.diag()>=tolWannier && cc.diag()<=1-tolWannier).eval().size();
            arma::mat kin=rot.tail_cols(len-nActive).t()*K*rot.tail_cols(len-nActive);
            arma::mat magicr=InactiveStarRotation(kin);
            auto givens=Fermionic::GivensRotForRot(magicr);
            for(auto &g : givens) g.b+=nActive;
//            auto givens=Fermionic::GivensRotForMatrix(cc,nExcludeGs,20);
            auto rot1=matrot_from_Givens(givens,cc.n_rows);
            //(rot1.t() * cc * rot1).print("rot1.t()*cc*rot1");
            auto gates=Fermionic::NOGates(sol.hamsys.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",1e-10,"Quiet",true, "DoNormalize",true});
            cout<<"circuit2:"<<t0.sincemark()<<endl;
            t0.mark();
            //cc=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //cc.print("cc after rot");
            // cout<<"nActive="<<nActive<<endl;
            // cout<<"rot to kin "<<magicr.n_rows<<endl;
            // cout<<"rot1 "<<rot1.n_rows<<endl;
            // cout.flush();

            rot = rot*rot1.t();
            cc=rot1*cc*rot1.t();
        }

        if (false && i%10==0) {// try Wannier of active orbitals:
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
