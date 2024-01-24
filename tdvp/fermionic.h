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

//    // fix the sign
//    for(auto j=0u; j<Q.n_cols; j++) {
//        auto i=arma::index_max(arma::abs(Q.col(j)));
//        Q.col(j) /= arma::sign(Q(i,j));
//    }

#ifndef NDEBUG
    double err=norm(A-Q*diagmat(eval2)*Q.t());
    std::cout<<"error diag RDR="<<norm(RDR-diagmat(eval))<<std::endl;
    std::cout<<"err eig_unitary="<<err<<std::endl;
    std::cout<<"err |eval|-1="<<norm(arma::abs(eval2)-ones(A.n_cols))<<std::endl;
#endif
    arma::mat J=arma::diagmat(arma::regspace(0,eval2.size()-1));
    arma::cx_mat X=Q.t() * J * Q;
    arma::vec Xeval=arma::real(X.diag());
    uvec Xiev=arma::stable_sort_index(Xeval);
    cx_vec lambda(A.n_rows, fill::ones);
    arma::cx_mat rot(A.n_rows,A.n_cols,fill::eye);
    rot.submat(nExclude,nExclude,A.n_rows-1,A.n_cols-1)=Q.cols(Xiev);
    lambda.rows(nExclude,A.n_rows-1)=eval2(Xiev);
    return make_pair(lambda,rot);
}

struct GivensRot {
    size_t b;     ///< bond b --- b+1
    double c, s, r;  ///< cos and sin and radius

    GivensRot(size_t b_) : b(b_) {}

    /// to eliminate p from (p,q). Adapted from eigen.tuxfamily.org
    GivensRot& make(double p, double q)
    {
        using Scalar=double;
        using std::sqrt;
        using std::abs;
        std::swap(p,q); // to eliminate the p instead of q.
        if(q==Scalar(0))
        {
            c = p<Scalar(0) ? Scalar(-1) : Scalar(1);
            s = Scalar(0);
            r = abs(p);
        }
        else if(p==Scalar(0))
        {
            c = Scalar(0);
            s = q<Scalar(0) ? Scalar(1) : Scalar(-1);
            r = abs(q);
        }
        else if(abs(p) > abs(q))
        {
            Scalar t = q/p;
            Scalar u = sqrt(Scalar(1) + t*t);
            if(p<Scalar(0))
                u = -u;
            c = Scalar(1)/u;
            s = -t * c;
            r = p * u;
        }
        else
        {
            Scalar t = p/q;
            Scalar u = sqrt(Scalar(1) + t*t);
            if(q<Scalar(0))
                u = -u;
            s = -Scalar(1)/u;
            c = -t * s;
            r = q * u;
        }
        return *this;
    }

    double angle() const { return atan2(s,c); }

    arma::mat22 matrix() const { return {{c,s},{-s,c}}; }
};

arma::mat matrot_from_Givens(std::vector<GivensRot> const& gates)
{
    size_t n=0;
    for(const GivensRot& g : gates) if (g.b>n) n=g.b;
    n+=2;
    arma::mat rot(n,n, arma::fill::eye);
    for(int i=gates.size()-1; i>=0; i--) { // apply to the right in reverse
//    for(auto i=0u; i<gates.size(); i++) {
        const GivensRot& g=gates[i];
        rot.submat(0,g.b,n-1,g.b+1) = rot.submat(0,g.b,n-1,g.b+1) * g.matrix();
    }
    return rot;
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
        itensor::Fermion sites(length(), {"ConserveNf=",true});
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

    // return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    static std::vector<GivensRot> NOGivensRot(arma::mat const& cc, int nExclude=2, size_t blockSize=8)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        std::vector<GivensRot> gs;
        arma::mat evec;
        arma::vec eval;
        size_t d=blockSize;
        for(auto p2=cc1.n_rows-1; p2>0u; p2--) {
            size_t p1= (p2+1>d) ? p2+1-d : 0u ;
            arma::mat cc2=cc1.submat(p1,p1,p2,p2);
            arma::eig_sym(eval,evec,cc2);
            // select the less active
            size_t pos=0;
            if (1-eval.back()<eval(0)) pos=eval.size()-1;
            arma::vec v=evec.col(pos);
            std::vector<GivensRot> gs1;
            for(auto i=0u; i+1<v.size(); i++)
            {
                auto b=i+p1;
                auto g=GivensRot(b).make(v[i],v[i+1]);
                gs1.push_back(g);
                v[i+1]=g.r;
            }
            auto rot1=matrot_from_Givens(gs1);
            cc1.submat(0,0,p2,p2)=rot1*cc1.submat(0,0,p2,p2)*rot1.t();
            for(auto g : gs1) { g.b+=nExclude; gs.push_back(g); }
        }
        return gs;
    }

    // return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    static std::vector<itensor::BondGate> NOGates(itensor::Fermion const& sites, std::vector<GivensRot> const& gs)
    {
        using itensor::BondGate;
        using itensor::Cplx_i;
        std::vector<itensor::BondGate> gates;
        for(const GivensRot& g : gs)
        {
            int b=g.b+1;
            auto hterm = ( sites.op("Adag",b)*sites.op("A",b+1)
                          -sites.op("Adag",b+1)*sites.op("A",b))* (g.angle()*Cplx_i);
            auto bg=BondGate(sites,b,b+1,BondGate::tReal,1,hterm);
            gates.push_back(bg);
        }
        return gates;
    }

    static arma::mat rotNO(arma::mat const& cc, int nExclude=2)
    {
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-14));
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

//    static arma::mat rotNO3(arma::mat const& cc, int nExclude=2, int nActive=8, double maxBlock=0)
    static arma::mat rotNO3(arma::mat const& cc, int nExclude=2, double tolWannier=1e-5, double maxBlock=0)
    {
        if (maxBlock==0) maxBlock=cc.n_rows;
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat J=arma::diagmat(arma::regspace(0,cc1.n_rows-1));
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);  // +1e-5/cc1.n_rows*J
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-14));
        eval(iev).print("evals");

        arma::vec eval2=eval(iev);
        arma::mat evec2=evec.cols(iev);

        // fix the sign and collect distance to 1
        arma::vec diff1(evec2.n_rows);
        {
            arma::mat one(evec2.n_rows, evec2.n_cols, fill::eye);
            for(auto j=0u; j<evec2.n_cols; j++) {
                auto i=arma::index_max(arma::abs(evec2.col(j)));
                evec2.col(j) /= arma::sign(evec2(i,j));
                diff1(j)=arma::norm(evec2.col(j)-one.col(j));
            }
        }

        // Wannier after activity sorting
        uvec weval;
        arma::vec Xeval;
        arma::mat Xevec;

        arma::vec x_sigma;
        {
            vec xi=(arma::mat {evec2.t()*J*evec2}).diag();
            vec xi2=(arma::mat {evec2.t()*(J-diagmat(xi)) * (J-diagmat(xi))*evec2}).diag();
            x_sigma=arma::sqrt(xi2.clean(1e-15));
        }
        //x_sigma.print("xsigma initial");

        {// group empty natural orbitals
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval2.size(); i++)
                if (eval2[i]<tolWannier ||
//                if ((i>=nActive && eval2[i]<0.5) ||
//                   (diff1[i]>1.0 && eval2[i]<0.5) ||
                   (2*x_sigma[i]>maxBlock && eval2[i]<0.5)) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec2.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::vec Xeval0;
            arma::mat Xevec0;
            arma::eig_sym(Xeval0,Xevec0,X);

            weval=ieval0;
            Xeval=Xeval0;
            Xevec=Xevec0;
        }
        {// group full natural orbitals
            std::vector<size_t> ieval1v;
            for(auto i=0u; i<eval2.size(); i++)
                if (std::abs(1.0-eval2[i])<tolWannier ||
//                if ((i>=nActive && eval2[i]>=0.5) ||
//                   (diff1[i]>1.0 && eval2[i]>=0.5) ||
                   (2*x_sigma[i]>maxBlock && eval2[i]>=0.5)) ieval1v.push_back(i);
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
            Xiev=arma::stable_sort_index(Xeval);
//            for(auto i=0u; i<Xeval.size(); i++) Xiev[i]=i;
//            int c=0;
//            for(auto i=0u; i<neval0; i++) Xiev[2*i]=i;
//            for(auto i=0; i+neval0<Xeval.size(); i++)
//                if (i<2*neval0) Xiev[2*i+1]=i+neval0;
//                else Xiev[2*neval0+c++]=i+neval0;
        }        

        arma::vec eval4=eval3;
        arma::mat evec4=evec3;
        eval4(arma::sort(weval))=eval3(weval(Xiev));
        evec4.cols(arma::sort(weval)) = evec3.cols(weval(Xiev));

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec4;

//        eval4.print("eval cicj");
//        Xeval(Xiev).print("orbitals position");

        {
            vec xi=(arma::mat {evec4.t()*J*evec4}).diag();
            vec xi2=(arma::mat {evec4.t()*(J-diagmat(xi)) * (J-diagmat(xi))*evec4}).diag();
            x_sigma=arma::sqrt(xi2.clean(1e-15));
            arma::join_horiz(xi,x_sigma).print("<X> sigmaX");
        }

        // fix the sign
        for(auto j=0u; j<rot.n_cols; j++) {
            auto i=arma::index_max(arma::abs(rot.col(j)));
            rot.col(j) /= arma::sign(rot(i,j));
        }

        std::cout<<"norm(1-rot)="<<arma::norm(rot-arma::mat(rot.n_rows, rot.n_cols, fill::eye))<< std::endl;
        if ( arma::norm(rot-arma::mat(rot.n_rows, rot.n_cols, fill::eye))>0.1 )
                       rot.print("rotation");

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

    static HamSys rotOp(arma::mat const& rot, int nExclude=2)
    {
        arma::mat rott=rot.t();
        arma::cx_mat logrot;
        const auto im=arma::cx_double(0,1);
        {
            auto [eval,evec]=eig_unitary(rott, nExclude);
            arma::cx_vec logeval=-arma::log(eval)*im;
            for(auto& x : logeval) { // fix sign of the
                if (std::real(x)>M_PI/2) x -= M_PI;
                else if (std::real(x)<-M_PI/2) x += M_PI;
            }
            logrot=evec*arma::diagmat(logeval)*evec.t();
            if (norm(logeval)>0.1) logeval.print("log(eval)");
            double err=arma::norm(rott-arma::expmat(im*logrot));
            if (err>1e-13) std::cout<<"exp error="<<err<<std::endl;

            double err_herm=norm(logrot.t()-logrot);
            if (err_herm>1e-13) std::cout<<"Hermitian error="<<err_herm<<std::endl;
        }
        arma::cx_mat kin=logrot; //arma::logmat(rott)*im; // we need to invert the rotation

        if (norm(kin)>0.1) kin.print("kin");
        auto L=rot.n_cols;
        itensor::Fermion sites(L, {"ConserveNf=",false});
        itensor::AutoMPO h(sites);
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (std::abs(kin(i,j))>1e-15)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return {sites, itensor::toMPO(h)};
    }

};

#endif // FERMIONIC_H
