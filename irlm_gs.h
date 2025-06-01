#ifndef IRLM_GS_H
#define IRLM_GS_H

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

struct Irlm_gs {
    IrlmData irlm;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    arma::sp_mat K;
    itensor::MPS psi;
    arma::sp_mat cc;
    int nActive=2;
    double tol=1e-10;


    explicit Irlm_gs(const IrlmData& irlm_, double tol_=1e-10)
        : irlm(irlm_)
        , sites(itensor::Fermion(irlm.L, {"ConserveNf",true}))
        , hImp (sites)
        , K(irlm_.star_kin())
        , tol(tol_)
    {
        prepareSlater(sites, arma::vec {K.diag()}, irlm.L/2);

        std::vector<int> impPos={0,1};
        auto Umat=irlm.U_mat();
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;
    }

    void extractRepresentative()
    {
        if (nActive+2 >= irlm.L) return; // there is no Slater
        arma::vec ni {cc.diag()};
        int p0=nActive;
        prepareSlater_2site(ni,p0);
        extract_f(ni,p0,true);
        extract_f(ni,p0,false);

        nActive+=2;
    }

    void prepareSlater(itensor::Fermion const& sites, arma::vec ek, int nPart)
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

private:
    /// the first two sites of the Slater will be |01> or |10>
    void prepareSlater_2site(arma::vec& ni, int p0)
    { // 1) find the first site where the occupation is not the one at p0
        int p1=p0+1;
        auto isEmpty=[](double x){ return x<0.5; };  // helper function
        for(; p1<irlm.L; p1++)
            if (isEmpty(ni[p1]) != isEmpty(ni[p0])) break;

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
    void extract_f(arma::vec const& ni, int p0, bool for_empty)
    {
        arma::vec nSlater=ni.rows(p0,irlm.L-1).eval();
        arma::uvec pos0= for_empty ?
                    arma::find(nSlater<0.5).eval()+p0 :
                    arma::find(nSlater>0.5).eval()+p0 ;
        if (pos0.empty()) return;
        auto k12 = K.rows(p0,p0+1).eval().cols(pos0);
        arma::vec s;
        arma::mat U, V;
        svd(U,s,V, arma::mat{k12});
    }
};

#endif // IRLM_GS_H
