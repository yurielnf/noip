#ifndef IRLM_GS_H
#define IRLM_GS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

struct IrlmData {
    int L=20;
    double t=0.5;
    double V=0.15;
    double U=-0.5;
    double ed=0;
    bool connected=true;

    /// Kinetic energy matrix
    arma::sp_mat kin_mat() const
    {
        arma::sp_mat K(L,L);
        for(auto i=1+!connected; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
        K(0,1)=K(1,0)=V;
        K(0,0)=ed;
        K(1,1)=-ed;

        K(1,1)=-U/2;
        K(0,0)+=-U/2;

        return K;
    }

    /// U ni nj
    arma::sp_mat U_mat() const
    {
        arma::sp_mat Umat(2,2);
        Umat(0,1)=U;
        return Umat;
    }

    /// Kinetic enery coeff in star geometry (Hbath is diagonal)
    arma::sp_mat star_kin() const
    {
        auto K=kin_mat();
        arma::mat Kbath=arma::mat {K.submat(2,2,L-1,L-1).eval()};
        arma::mat evec;
        arma::vec ek;
        arma::eig_sym(ek,evec,Kbath);
        arma::vec vk=(K.submat(1,2,1,L-1)*evec).as_col();

        arma::sp_mat Kstar(L,L);
        Kstar.submat(0,0,1,1)=K.submat(0,0,1,1);
        for(auto i=0u;i<ek.size();i++) {
            Kstar(i+2,i+2)=ek[i];
            Kstar(1,i+2)=vk[i];
        }
        return Kstar;
    }
};

struct DmrgParams {
    int max_bond_dim=512;
    int nIter_diag=4;
    double noise=1e-8;
};

struct Irlm_gs {
    IrlmData irlm;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    double tol=1e-10;

    /// these quantities are updated during the iterations
    arma::mat K;
    itensor::MPS psi;
    double energy;
    arma::mat cc;
    int nActive=2;


    explicit Irlm_gs(const IrlmData& irlm_, double tol_=1e-10)
        : irlm(irlm_)
        , sites(itensor::Fermion(irlm.L, {"ConserveNf",true}))
        , hImp (sites)
        , K(irlm_.star_kin())
        , tol(tol_)
    {
        prepareSlaterGs(sites, arma::vec {K.diag()}, irlm.L/2);

        std::vector<int> impPos={0,1};
        auto Umat=irlm.U_mat();
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;
    }

    void iterate(DmrgParams args={})
    {
        extractRepresentative();
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    void extractRepresentative()
    {
        if (nActive+2 >= irlm.L) return; // there is no Slater
        arma::vec ni {cc.diag()};
        int p0=nActive;
        reorderSlater_2site(ni,p0);
        extract_f(ni,p0,0.0);
        extract_f(ni,p0,1.0);
        nActive+=2;
    }

    void doDmrg(DmrgParams args={})
    {
        auto h=hImp;
        for(auto i=0; i<nActive; i++)
            for(auto j=0; j<nActive; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);

        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(psi,mpo,sweeps, {"Quiet", true, "Silent", true});
        energy += SlaterEnergy();

        auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
    }

    void rotateToNaturalOrbitals()
    {
        auto cc1=cc.submat(2,2,nActive-1, nActive-1).eval();
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=2;
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        K.cols(0,nActive-1)=K.cols(0,nActive-1).eval()*rot1.st();
        K.rows(0,nActive-1)=rot1.st().t()*K.rows(0,nActive-1).eval();
        auto nib=arma::real(cc.diag()).eval().rows(2,irlm.L-1).eval();
        nActive=arma::find(nib>tol && nib<1-tol).eval().size()+2;
    }

    void prepareSlaterGs(itensor::Fermion const& sites, arma::vec ek, int nPart)
    {
        cc=arma::sp_mat(irlm.L, irlm.L);
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        double energy=0;
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            energy += ek[k];
            cc(k,k)=1;
        }
        std::cout << " Slater energy: " << energy << std::endl;
        psi=itensor::MPS(state);
    }

    double SlaterEnergy()
    {
        double energy=0;
        for(auto i=nActive; i<irlm.L; i++)
            energy += cc(i,i)*K(i,i);
        return energy;
    }

    itensor::MPO fullHamiltonian() const
    {
        auto h=hImp;
        for(auto i=0; i<irlm.L; i++)
            for(auto j=0; j<irlm.L; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }

private:
    /// the first two sites of the Slater will be |01> or |10>
    void reorderSlater_2site(arma::vec& ni, int p0)
    { // 1) find the first site where the occupation is not the one at p0
        int p1=p0+1;
        for(; p1<irlm.L; p1++)
            if (std::abs(ni[p1]-ni[p0])>0.5) break;

        if (p1 != p0+1 && p1<irlm.L){ // 1a) swap sites in the Hamiltonian and in the state
            K.swap_cols(p1,p0+1);
            K.swap_rows(p1,p0+1);

            cc.swap_cols(p1,p0+1);
            cc.swap_rows(p1,p0+1);
            std::swap(ni[p1],ni[p0+1]);

            itensor::AutoMPO ampo(sites);
            ampo += "Cdag",p1+1,"C",p0+2;
            ampo += "Cdag",p0+2, "C",p1+1;
            auto op = itensor::toMPO(ampo);
            psi = applyMPO(op,psi);
            psi.replaceSiteInds(sites.inds());
        }
    }

    /// extract f orbital of the sites with ni=0 or 1
    void extract_f(arma::vec const& ni, int p0, double nRef)
    {
        arma::vec nSlater=arma::abs(ni.rows(p0,irlm.L-1)-nRef).eval();
        arma::uvec pos0=arma::find(nSlater<0.5).eval()+p0 ;
        if (pos0.empty()) return;
        auto k12 = K.rows(p0,p0+1).eval().cols(pos0).eval();
        arma::vec s;
        arma::mat U, V;
        svd(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        if (nSv>1) std::cout<<"nSV="<<nSv<<std::endl;
        auto givens=GivensRotForRot_left(arma::conj(V.head_cols(nSv)).eval());
        for(auto& g:givens) g.b+=p0;
        GivensDaggerInPlace(givens);
        applyGivens(K,givens);
        applyGivens(GivensDagger(givens),K);
        // no need to update cc
    }
};

#endif // IRLM_GS_H
