#ifndef FERMIONIC_H
#define FERMIONIC_H

#include<armadillo>
#include <map>
#include <array>
#include <itensor/all.h>

struct Fermionic {
    arma::mat Kmat, Umat;
    std::map<std::array<int,4>, double> Vijkl;
    arma::mat Rot;

    Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_, std::map<std::array<int,4>, double> const& Vijkl_={})
        :Kmat(Kmat_), Umat(Umat_), Vijkl(Vijkl_)
    {}

    int length() const { return Kmat.n_rows; }

    itensor::AutoMPO Kin() const
    {
        int L=length();
        itensor::AutoMPO h;
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (fabs(Kmat(i,j))>1e-15)
                    h += Kmat(i,j),"Cdag",i+1,"C",j+1;
        return h;
    }

    itensor::AutoMPO Interaction() const
    {
        if (!Rot.empty()) return InteractionRot();
        int L=length();
        itensor::AutoMPO h;
        // Uij ni nj
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (fabs(Umat(i,j))>1e-15)
                    h += Umat(i,j),"Cdag",i+1,"C",i+1,"Cdag",j+1,"C",j+1;

        for(const auto& it : Vijkl)
        {
            auto pos=it.first; // i, j, k, l
            auto coeff=it.second;
            if (fabs(coeff)>1e-15)
                h += coeff,"Cdag",pos[0]+1,"C",pos[1]+1,"Cdag",pos[2]+1,"C",pos[3]+1;
        }
        return h;
    }

    itensor::AutoMPO InteractionRot(double tol=1e-14) const
    {
        int L=length();
        itensor::AutoMPO h;
        // Uij ni nj
        for(int i=0;i<Rot.n_rows; i++)
            for(int j=0;j<Rot.n_rows; j++)
            {
//                if (fabs(Umat(i,j))<tol) continue;

//                auto ci=MPSSum(2,MatSVDFixedTol(tol));
//                for(int a=0;a<L; a++)
//                    ci += Destroy(a) * Rot(i,a);

//                auto cid=MPSSum(2,MatSVDFixedTol(tol));
//                for(int a=0;a<L; a++)
//                    cid += Create(a) * Rot(i,a);

//                auto cj=MPSSum(2,MatSVDFixedTol(tol));
//                for(int a=0;a<L; a++)
//                    cj += Destroy(a) * Rot(j,a);

//                auto cjd=MPSSum(2,MatSVDFixedTol(tol));
//                for(int a=0;a<L; a++)
//                    cjd += Create(a) * Rot(j,a);

//                h += cid.toMPS() * ci.toMPS() * cjd.toMPS() * cj.toMPS() * Umat(i,j);
            }

        return h;
    }


    itensor::MPO Ham() const
    {
        itensor::AutoMPO h = Kin();
        h += Interaction();
        return itensor::toMPO(h);
    }
};

#endif // FERMIONIC_H
