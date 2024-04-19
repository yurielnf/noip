#ifndef IRLM_H
#define IRLM_H


#include "fermionic.h"
#include <tuple>

struct IRLM {
    int L=20;
    double t=0.5;
    double V=0.15;
    double U=-0.5;
    double ed=0;

    auto matrices() const
    {
        // Kinetic energy TB Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
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

    /// the Hamiltonian setting the kinetic(inactive,inactive)=0
    HamSys HamRestricted(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        auto hsys=sys.Ham();
        hsys.hamEnrich=hsys.ham;
        return hsys;
    }

    HamSys Ham(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        HamSys hsys=sys.Ham();
        if (iInactive.empty()) return hsys;
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        hsys.hamEnrich=sys.Ham().ham;
        return hsys;
    }

    HamSysV HamV(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        HamSysV hsys=sys.HamV();
        if (iInactive.empty()) return hsys;  // normal path
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        hsys.hamEnrich=sys.Ham().ham;
        return hsys;
    }

    HamSys HamStar() const { return Ham(rotStar(), true); }

    HamSys HamNO(arma::mat const& cc, bool fromStar=true) const { auto R=Fermionic::rotNO(cc); if (fromStar) R=R*rotStar(); return Ham(R); }

};


#endif // IRLM_H
