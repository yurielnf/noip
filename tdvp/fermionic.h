#ifndef FERMIONIC_H
#define FERMIONIC_H

#include<armadillo>
#include <map>
#include <array>
#include <itensor/all.h>

auto eig_unitary(const arma::mat& A)
{
    using namespace arma;
    cx_vec eval, eval2;
    cx_mat evec, Q, R;
    eig_gen(eval, evec, A);
    qr(Q, R, evec);
    cx_mat RDR=R*diagmat(eval)*R.i();
    eval2=RDR.diag();
    std::cout<<"error diag RDR="<<norm(RDR-diagmat(eval))<<std::endl;
    double err=norm(A-Q*diagmat(eval2)*Q.t());
    std::cout<<"err eig_unitary="<<err<<std::endl;
    std::cout<<"err |eval|-1="<<norm(arma::abs(eval2)-ones(A.n_cols))<<std::endl;
    return make_pair(eval2,Q);
}


struct HamSys {
    itensor::Fermion sites;
    itensor::MPO ham;
};

struct HamSysExact {
    itensor::Fermion sites;
    itensor::AutoMPO ampo;
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
//                                if (fabs(Umat(i,j))>1e-15)
                                    Vabcd += Umat(i,j)*Rot(i,a)*Rot(i,b)*Rot(j,c)*Rot(j,d);
                        if (fabs(Vabcd)>1e-15)
                            h += Vabcd,"Cdag",a+1,"C",b+1,"Cdag",c+1,"C",d+1;
                    }
    }

    HamSys Ham() const
    {
        itensor::Fermion sites(length(), {"ConserveNf=",false});
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
        arma::vec eval2(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            eval2[i]=-std::min(eval[i], -eval[i]+1);
        arma::uvec iev=arma::sort_index(eval2);
        eval(iev).print("evals");
        return evec.cols(iev);
    }

    static arma::mat rotNO2(arma::mat const& cc, double tolWannier=1e-5)
    {
        using namespace arma;
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc);
        arma::vec eval2(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            eval2[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::sort_index(eval2);
        eval(iev).print("evals");

        arma::mat evec2=evec.cols(iev);
        arma::mat evec3=evec2;

        // Wannier after activity sorting
        arma::mat J=arma::diagmat(arma::regspace(0,eval.size()-1));
        {
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval.size(); i++)
                if (eval[iev[i]]<tolWannier) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec2.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::mat wevec;
            arma::vec weval;
            arma::eig_sym(weval,wevec,X);
            evec3.cols(ieval0) = evec2.cols(ieval0) * wevec;
        }
        arma::mat evec4=evec3;
        {
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval.size(); i++)
                if (std::abs(1.0-eval[iev[i]])<tolWannier) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec3.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::mat wevec;
            arma::vec weval;
            arma::eig_sym(weval,wevec,X);
            evec4.cols(ieval0) = evec3.cols(ieval0) * wevec;
        }

        return evec4;
    }

    static HamSysExact rotOpExact(arma::mat const& rot)
    {
        arma::mat rott=rot.t();
        arma::cx_mat logrot;
        const auto im=arma::cx_double(0,1);
        {
            arma::cx_mat evec;
            arma::cx_colvec eval;
            arma::eig_gen(eval,evec,rott);
            logrot=evec*arma::diagmat(arma::log(eval)*im)*evec.i();
        }
        arma::cx_mat kin=logrot; //arma::logmat(rott)*im; // we need to invert the rotation
        auto L=rot.n_cols;
        itensor::Fermion sites(L, {"ConserveNf=",true});
        itensor::AutoMPO h(sites);
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (std::abs(kin(i,j))>1e-15)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        arma::imag(kin).print("Hrot");
        return {sites, h};
    }

    static HamSys rotOp(arma::mat const& rot)
    {
        arma::mat rott=rot.t();
        arma::cx_mat logrot;
        const auto im=arma::cx_double(0,1);
        {
            auto [eval,evec]=eig_unitary(rott);
            logrot=evec*arma::diagmat(arma::log(eval)*im)*evec.t();

            double err=arma::norm(rott-arma::expmat(-im*logrot));
            if (err>1e-13) std::cout<<"exp error="<<err<<std::endl;

            double err_herm=norm(logrot.t()-logrot);
            if (err_herm>1e-13) std::cout<<"Hermitian error="<<err_herm<<std::endl;
        }
        arma::cx_mat kin=logrot; //arma::logmat(rott)*im; // we need to invert the rotation
        auto L=rot.n_cols;
        itensor::Fermion sites(L, {"ConserveNf=",true});
        itensor::AutoMPO h(sites);
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (std::abs(kin(i,j))>1e-15)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return {sites, itensor::toMPO(h)};
    }

};

#endif // FERMIONIC_H
