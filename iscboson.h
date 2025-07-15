#ifndef ISCBOSON_H
#define ISCBOSON_H

#include <itensor/all.h>
#include <armadillo>

struct HamBSys {
    itensor::Boson sites;
    itensor::MPO ham;
};


struct IscbParam {
    int L=20;
    double wp=1.0;          ///< plasma frequency
    double alpha=0.1;       ///< hybridization prefactor
    double eJ=0.1;          ///< impurity
    int maxOcc=3;           ///< bound to the occupation number of bosons for each site

    auto QuadraticMat() const
    {
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=0; i<L; i++) {
            K(i,i)=wp*sin(M_PI*(i+1)/(2*L));
            if (i>0)
                K(0,i)=K(i,0)=alpha*sqrt(i/L);
        }
        return K;
    }
};

struct Iscb {
    IscbParam param;
    itensor::Boson sites;
    itensor::AutoMPO hImp;
    arma::mat K;

    explicit Iscb(const IscbParam& param_)
        : param(param_)
        , sites(param_.L, {"MaxOcc=",param_.maxOcc,"ConserveQNs=",false})
        , hImp (sites)
    {
        K=param_.QuadraticMat();
        hImp += 0.5*param_.eJ, "A*A",1;
        hImp += 0.5*param_.eJ, "Adag*Adag",1;
        hImp += 0.5*param_.eJ, "A*Adag",1;
        hImp += 0.5*param_.eJ, "Adag*A",1;
    }

    HamBSys Ham() const
    {
        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(K(i,j))>1e-15)
                h += K(i,j),"Adag",i+1,"A",j+1;
        auto mpo=itensor::toMPO(h);
        return {sites,mpo};
    }
};





#endif // ISCBOSON_H
