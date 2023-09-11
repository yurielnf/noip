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

    static arma::mat cc_matrix(itensor::MPS const& gs, itensor::Fermion const& sites)
    {
        auto ccz=correlationMatrixC(gs, sites,"Cdag","C");
        arma::mat cc(ccz.size(), ccz.size());
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=std::real(ccz[i][j]);
        return cc;
    }

    static arma::mat rotNO(arma::mat const& cc)
    {
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc);
        std::cout<<"NO eigen error="<<arma::norm(cc-evec*arma::diagmat(eval)*evec.t())<<std::endl;
        arma::vec eval2(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            eval2[i]=-std::min(eval[i], -eval[i]+1);
        auto iev=arma::sort_index(eval2);
        eval(iev).print("evals");
        return evec.cols(iev);
    }

    static HamSys rotOp(arma::mat const& rot)
    {
        const auto im=arma::cx_double(0,1);
        arma::cx_mat kin=arma::logmat(rot.t())*im; // we need to invert the rotation
        {
            std::cout<<"norm(hrot)="<<arma::norm(arma::real(kin))<<" ";
            std::cout<<arma::norm(arma::imag(kin))<<"\n";
            std::cout.flush();
        }
        std::cout<<"rot unitary error="<<arma::norm(rot.t()*rot-arma::eye(arma::size(rot)))<<std::endl;
        std::cout<<"log(rot) Hermitian error="<<arma::norm(kin-kin.t())<<std::endl;
        std::cout<<"exp error="<<arma::norm(arma::expmat(-im*kin)-rot.t())<<std::endl;
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
