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


    explicit Irlm_gs(itensor::Fermion sites_, const IrlmData& irlm_)
        : irlm(irlm_)
        , sites(sites_)
        , hImp (sites)
        , K(irlm_.star_kin())
        , psi(prepareSlater(sites, arma::vec {K.diag()}, irlm.L/2))
    {
        std::vector<int> impPos={0,1};
        auto Umat=irlm.U_mat();
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;
    }

    static itensor::MPS prepareSlater(itensor::Fermion const& sites, arma::vec ek, int nPart)
    {
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        double energy=0;
        for(int j = 0; j < nPart; j++) {
            state.set(iek[j]+1,"1");
            energy += ek[iek[j]];
        }
        std::cout << " Slater energy: " << energy << std::endl;
        return itensor::MPS(state);
    }

};

#endif // IRLM_GS_H
