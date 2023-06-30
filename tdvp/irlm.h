#ifndef IRLM_H
#define IRLM_H


#include "fermionic.h"

struct IRLM {
    int L=20;
    double t=0.5;
    double V=0.15;
    double U=-0.5;

    auto matrices() const
    {
        // Kinetic energy TB Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
        K(0,1)=K(1,0)=V;

        // U ni nj
        arma::mat Umat(L,L,arma::fill::zeros);
        Umat(0,1)=U;
        K(0,0)=K(1,1)=-U/2;
        return std::make_pair(K,Umat);
    }

    auto matricesStar() const
    {
        auto [K,Umat]=matrices();
        { // Diagonalize the bath
            arma::vec ek;
            arma::mat R;
            arma::eig_sym( ek, R, K.submat(2,2,L-1,L-1) );
            arma::uvec iek=arma::sort_index( arma::abs(ek) );
            arma::mat Rfull(L,L,arma::fill::eye);
            arma::mat Rr=R.cols(iek);
            Rfull.submat(2,2,L-1,L-1)=Rr;
            K=Rfull.t()*K*Rfull;
        }
        return std::make_pair(K,Umat);
    }

    HamSys Ham() const
    {
        auto [K,Umat]=matrices();
        return Fermionic(K,Umat).Ham();
    }

    HamSys HamStar() const
    {
        auto [K,Umat]=matricesStar();
        return Fermionic(K,Umat).Ham();
    }

    HamSys HamNO(itensor::VecVecR const& cc) const
    {
        arma::mat evec, ccm(cc.size(),cc.size());
        arma::vec eval;
        for(auto i=0u; i<ccm.n_rows; i++)
            for(auto j=0u; j<ccm.n_cols; j++)
                ccm(i,j)=cc[i][j];
        arma::eig_sym(eval,evec,ccm);
        auto [K,Umat]=matricesStar();
        return Fermionic(K,Umat,evec).Ham();
    }

    static itensor::VecVecR cc_matrix(itensor::MPS const& gs, itensor::Fermion const& sites) { return Fermionic::cc_matrix(gs,sites); }

};


#endif // IRLM_H
