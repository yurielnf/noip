#ifndef FERMIONIC_H
#define FERMIONIC_H

#include<armadillo>
#include <map>
#include <array>
#include <itensor/all.h>

auto eig_unitary(const arma::mat& A, int nExclude=2)
{
    using namespace arma;
    arma::mat A1=A.submat(nExclude,nExclude,A.n_rows-1,A.n_cols-1);
    cx_vec eval, eval2;
    cx_mat evec, Q, R;
    eig_gen(eval, evec, A1);
    qr(Q, R, evec);
    cx_mat RDR=R*diagmat(eval)*R.i();
    eval2=RDR.diag();
#ifndef NDEBUG
    double err=norm(A-Q*diagmat(eval2)*Q.t());
    std::cout<<"error diag RDR="<<norm(RDR-diagmat(eval))<<std::endl;
    std::cout<<"err eig_unitary="<<err<<std::endl;
    std::cout<<"err |eval|-1="<<norm(arma::abs(eval2)-ones(A.n_cols))<<std::endl;
#endif
    cx_vec lambda(A.n_rows, fill::ones);
    arma::cx_mat rot(A.n_rows,A.n_cols,fill::eye);
    rot.submat(nExclude,nExclude,A.n_rows-1,A.n_cols-1)=Q;
    lambda.rows(nExclude,A.n_rows-1)=eval2;
    return make_pair(lambda,rot);
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

    static arma::mat rotNO(arma::mat const& cc, int nExclude=2)
    {
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
        arma::vec eval2(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            eval2[i]=-std::min(eval[i], -eval[i]+1);
        arma::uvec iev=arma::sort_index(eval2);
        eval(iev).print("evals");
        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec.cols(iev);
        return rot;
    }

    static arma::mat rotNO2(arma::mat const& cc, int nExclude=2, double tolWannier=1e-5)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
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

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec4;
        return rot;
    }

    static arma::mat rotNO3(arma::mat const& cc, int nExclude=2, double tolWannier=1e-5)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-14));
        eval(iev).print("evals");

        arma::vec eval2=eval(iev);
        arma::mat evec2=evec.cols(iev);

        // Wannier after activity sorting
        arma::mat J=arma::diagmat(arma::regspace(0,eval.size()-1));
        size_t neval0;
        uvec weval;
        arma::vec Xeval;
        arma::mat Xevec;

        {// group empty natural orbitals
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval2.size(); i++)
                if (eval2[i]<tolWannier) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec2.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::vec Xeval0;
            arma::mat Xevec0;
            arma::eig_sym(Xeval0,Xevec0,X);

            weval=ieval0;
            Xeval=Xeval0;
            Xevec=Xevec0;
            neval0=weval.size();
        }
        {// group full natural orbitals
            std::vector<size_t> ieval1v;
            for(auto i=0u; i<eval.size(); i++)
                if (std::abs(1.0-eval[iev[i]])<tolWannier) ieval1v.push_back(i);
            uvec ieval1=conv_to<uvec>::from(ieval1v);
            arma::mat evec1=evec2.cols(ieval1);
            arma::mat X=evec1.t()* J * evec1;
            arma::vec Xeval1;
            arma::mat Xevec1;
            arma::eig_sym(Xeval1,Xevec1,X);

            weval=join_vert(weval,ieval1);
            Xeval=join_vert(Xeval,Xeval1);
            arma::mat rot(Xeval.size(), Xeval.size(), fill::zeros); // tensor addition of the two rotations
            rot.submat(0,0,Xevec.n_rows-1,Xevec.n_cols-1)=Xevec;
            rot.submat(Xevec.n_rows,Xevec.n_cols,rot.n_rows-1,rot.n_cols-1)=Xevec1;
            Xevec=rot;
        }

        // apply Wannierization
        arma::vec eval3=eval2;
        arma::mat evec3=evec2;
        evec3.cols(weval) = evec2.cols(weval) * Xevec;

        // sort Wanier orbitals according to position
        arma::uvec Xiev(Xeval.size());
        {
            Xiev=arma::sort_index(Xeval);
            for(auto i=0u; i<Xeval.size(); i++) Xiev[i]=i;
//            int c=0;
//            for(auto i=0u; i<neval0; i++) Xiev[2*i]=i;
//            for(auto i=0; i+neval0<Xeval.size(); i++)
//                if (i<2*neval0) Xiev[2*i+1]=i+neval0;
//                else Xiev[2*neval0+c++]=i+neval0;
        }
        Xeval.print("orbitals position");

        arma::vec eval4=eval3;
        arma::mat evec4=evec3;
        eval4(weval)=eval3(weval(Xiev));
        evec4.cols(weval) = evec3.cols(weval(Xiev));

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec4;
        return rot;
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
