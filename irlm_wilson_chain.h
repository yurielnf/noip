#ifndef IRLM_WILSON_CHAIN_H
#define IRLM_WILSON_CHAIN_H

#include "fermionic.h"
#include <tuple>

struct IRLM_wilson_chain {
    int L=30;
    double t0=0.5;
    double Lambda=1.5;
    double V=0.15;
    double U=-0.5;
    double ed=0;

    auto matrices() const
    {
        // Kinetic energy Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto n=0; n<L-2; n++) {
            double a=(1+pow(Lambda,-1)) * (1-pow(Lambda,-n-1)) * pow(Lambda,-0.5*n);
            double b=(1-pow(Lambda,-2*n-1))*(1-pow(Lambda,-2*n-3));
            K(n+1,n+2)=K(n+2,n+1)=t0*0.5*a/sqrt(b);
        }
        K(0,1)=K(1,0)=V;
        K(0,0)=ed;

        // U ni nj
        arma::mat Umat(L,L,arma::fill::zeros);
        Umat(0,1)=U;
        K(1,1)=-U/2;
        K(0,0)+=-U/2;
        return std::make_pair(K,Umat);
    }

    auto rotStar() const
    {
        auto [K, _]=matrices();
        // Diagonalize the bath
        arma::vec ek;
        arma::mat R;
        arma::eig_sym( ek, R, K.submat(2,2,L-1,L-1) );
        arma::uvec iek=arma::sort_index( arma::abs(ek) );
        arma::mat Rfull(L,L,arma::fill::eye);
        arma::mat Rr=R.cols(iek);
        Rfull.submat(2,2,L-1,L-1)=Rr;
        return Rfull;
    }

    HamSys Ham(arma::mat const& rot={}, bool rotateOnlyKin=false) const
    {
        auto [K,Umat]=matrices();
        if (rot.empty()) return Fermionic(K,Umat).Ham();
        if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat).Ham();
        return Fermionic(K,Umat,rot).Ham();
    }

    HamSys HamStar() const { return Ham(rotStar(), true); }

    HamSys HamNO(arma::mat const& cc, bool fromStar=true) const { auto R=Fermionic::rotNO(cc); if (fromStar) R=R*rotStar(); return Ham(R); }

};


#endif // IRLM_WILSON_CHAIN_H
