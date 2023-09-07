#ifndef FERMIONIC_H
#define FERMIONIC_H

#include<armadillo>
#include <map>
#include <array>
#include <itensor/all.h>

struct HamSys {
    itensor::Fermion sites;
    itensor::MPO ham;
};

struct Fermionic {
    arma::mat Kmat, Umat;
    std::map<std::array<int,4>, double> Vijkl;
    arma::mat Rot;


    explicit Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_={}, std::map<std::array<int,4>, double> const& Vijkl_={})
        : Kmat(Kmat_), Umat(Umat_), Vijkl(Vijkl_)
    {}

    Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_,
              arma::mat const& Rot_, bool rotateKin=true)
        : Kmat(rotateKin ? Rot_.t()*Kmat_*Rot_ : Kmat_)
        , Umat(Umat_), Rot(Rot_)
    {}

    int length() const { return Kmat.n_rows; }

    void Kin(itensor::AutoMPO& h) const
    {
        int L=length();
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (fabs(Kmat(i,j))>1e-15)
                    h += Kmat(i,j),"Cdag",i+1,"C",j+1;
    }

    void Interaction(itensor::AutoMPO& h) const
    {
        if (Umat.empty() && Vijkl.empty()) return;
        if (!Rot.empty()) return InteractionRot(h);
        // Uij ni nj
        for(int i=0;i<Umat.n_rows; i++)
            for(int j=0;j<Umat.n_cols; j++)
                if (fabs(Umat(i,j))>1e-15)
                    h += Umat(i,j),"Cdag",i+1,"C",i+1,"Cdag",j+1,"C",j+1;

        for(const auto& [pos,coeff] : Vijkl)
            if (fabs(coeff)>1e-15)
                h += coeff,"Cdag",pos[0]+1,"C",pos[1]+1,"Cdag",pos[2]+1,"C",pos[3]+1;
    }

    void InteractionRot(itensor::AutoMPO& h) const
    {
        int L=length();
        // Uij ni nj
        for(auto a=0; a<L; a++)
            for(auto b=0; b<L; b++)
                for(auto c=0; c<L; c++)
                    for(auto d=0; d<L; d++) {
                        double Vabcd=0;
                        for(int i=0;i<Umat.n_rows; i++)
                            for(int j=0;j<Umat.n_cols; j++)
                                if (fabs(Umat(i,j))>1e-15)
                                    Vabcd += Umat(i,j)*Rot(a,i)*Rot(b,i)*Rot(c,j)*Rot(d,j);
                        if (fabs(Vabcd)>1e-15)
                            h += Vabcd,"Cdag",a+1,"C",b+1,"Cdag",c+1,"C",d+1;
                    }
    }

    HamSys Ham() const
    {
        itensor::Fermion sites(length(), {"ConserveQNs=",false});
        itensor::AutoMPO h(sites);
        Kin(h);
        Interaction(h);
        return {sites, itensor::toMPO(h)};
    }

    static itensor::VecVecR cc_matrix(itensor::MPS const& gs, itensor::Fermion const& sites)
    {
        auto ccz=correlationMatrixC(gs, sites,"Cdag","C");
        itensor::VecVecR cc(ccz.size());
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc[i].push_back( std::real(ccz[i][j]) );
        return cc;
    }

    static auto rotNO(itensor::VecVecR const& cc)
    {
        arma::mat evec, ccm(cc.size(),cc.size());
        arma::vec eval;
        for(auto i=0u; i<ccm.n_rows; i++)
            for(auto j=0u; j<ccm.n_cols; j++)
                ccm(i,j)=std::real(cc[i][j]);
        arma::eig_sym(eval,evec,ccm);
        eval.print("evals");
        return evec;
    }

    static HamSys rotOp(arma::mat const& rot)
    {
        arma::cx_mat kin=arma::logmat(rot)*arma::cx_double(0,1);
        {
            std::cout<<arma::norm(arma::real(kin))<<" ";
            std::cout<<arma::norm(arma::imag(kin))<<"\n";
            std::cout.flush();
        }
        auto L=rot.n_cols;
        itensor::Fermion sites(L, {"ConserveQNs=",false});
        itensor::AutoMPO h(sites);
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (std::abs(kin(i,j))>1e-15)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return {sites, itensor::toMPO(h)};
    }

};

#endif // FERMIONIC_H
