#ifndef IRLM_H
#define IRLM_H


#include "fermionic.h"
#include <tuple>

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

    static auto rotNO(itensor::VecVecR const& cc)
    {
        arma::mat evec, ccm(cc.size(),cc.size());
        arma::vec eval;
        for(auto i=0u; i<ccm.n_rows; i++)
            for(auto j=0u; j<ccm.n_cols; j++)
                ccm(i,j)=std::real(cc[i][j]);
        arma::eig_sym(eval,evec,ccm);
        return evec;
    }

    static auto rotOp(arma::mat const& rot)
    {
        arma::mat h=-arma::imag(arma::logmat(rot));
        std::cout<<arma::norm(arma::real(arma::logmat(rot)));
        std::cout<<arma::norm(arma::imag(arma::logmat(rot)));
        std::cout.flush();
        return Fermionic(h).Ham();
    }

    HamSys HamStar() const { return Ham(rotStar(), true); }

    HamSys HamNO(itensor::VecVecR const& cc, bool fromStar=true) const { auto R=rotNO(cc); if (fromStar) R=R*rotStar(); return Ham(R); }

    static itensor::VecVecR cc_matrix(itensor::MPS const& gs, itensor::Fermion const& sites) { return Fermionic::cc_matrix(gs,sites); }



};


#endif // IRLM_H
