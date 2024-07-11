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
    bool connected=true;

    auto matrices() const
    {
        // Kinetic energy TB Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=1+!connected; i<L-1; i++)
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
        hsys.hamEnrich=hsys.ham;
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

struct IRLM_ip {
    IRLM irlm;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    arma::mat K;

    explicit IRLM_ip(const IRLM& irlm_)
        : irlm(irlm_)
        , sites(irlm_.L, {"ConserveNf",true})
        , hImp (sites)
    {
        arma::mat Umat;
        std::tie(K,Umat)=irlm.matrices();

        std::vector<int> impPos={0,1};
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;
    }

    template<class T>
    HamSys Ham(arma::Mat<T> const& rot) const
    {
        arma::cx_mat Kin=rot.t()*K*rot;
        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kin(i,j))>1e-15)
                h += Kin(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        return {sites,mpo,mpo};
    }

    template<class T>
    HamSys HamIP(arma::Mat<T> const& rot, int nImp, double dt) const
    {
        if (nImp==rot.n_rows) return Ham(rot);
        arma::Mat<T> K0=rot.t()*K*rot;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                           K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt);
        arma::cx_mat Kip=K0 * arma::cx_double(1,0);
        Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
        Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
        Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
        Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>1e-15)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        return {sites,mpo,mpo};
    }

    /// return exp(-i H2 dt), i.e. the rotation generated after the interaction picture evolution
    template<class T>
    arma::cx_mat rotIP(arma::Mat<T> const& rot, int nImp, double dt)
    {
        using namespace arma;
        arma::Mat<T> Kin=rot.t()*K*rot;
        arma::cx_mat rotK(size(rot), fill::eye);
        rotK.submat(nImp,nImp, rot.n_rows-1,rot.n_rows-1)=expIH<T>(Kin.submat(nImp,nImp, rot.n_rows-1,rot.n_rows-1) * dt );
        return rotK;
    }

    /// return the HamSys and the list of Givens rotations.
    template<class T>
    std::tuple<HamSys, std::vector<GivensRot<T>>, arma::Mat<T> > HamIP_f(arma::Mat<T> const& rot, int nImp, double dt, double tolSv=1e-12) const
    {
        if (nImp==rot.n_rows) return {Ham(rot),{},{}};
        arma::Mat<T> K0=rot.t()*K*rot;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                           K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt); //the commutator
        arma::cx_mat Kip=K0 * arma::cx_double(1,0);
        Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
        Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
        Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
        Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

        std::vector<GivensRot<T>> givens;
        {// the circuit to extract f orbitals
            auto k12=Kip.submat(0,nImp,nImp-1,rot.n_rows-1);
            arma::vec s;
            arma::Mat<T> U, V;
            svd_econ(U,s,V,arma::conj(k12));
            int nSv=arma::find(s>tolSv*s[0]).eval().size();
            std::cout<<"nSV="<<nSv<<std::endl;
            givens=GivensRotForRot_left(V.head_cols(nSv).eval());
            for(auto& g:givens) g.b+=nImp;
            arma::Mat<T> rot1=matrot_from_Givens(givens, rot.n_cols).st();
            Kip=(rot1.t()*Kip*rot1).eval();
        }

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>1e-15)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        HamSys ham{sites,mpo,mpo};
        return make_tuple(ham, givens, Kip);
    }

    template<class T>
    auto TrotterGates(arma::Mat<T> const& Kip,int nTB,double dt) const
    {
        using namespace itensor;

        auto gates = std::vector<BondGate>();

        //Create the gates exp(-i*tstep/2*hterm)
        for(int b = 1; b <= nTB-1; ++b)
        {
            auto hterm = Kip(b-1,b)*op(sites,"Adag",b)*op(sites,"A",b+1);
            hterm += Kip(b,b-1)*op(sites,"A",b)*op(sites,"Adag",b+1);
            hterm += Kip(b-1,b-1)*op(sites,"N",b)*op(sites,"Id",b+1);
            hterm += Kip(b,b)*op(sites,"Id",b)*op(sites,"N",b+1);
            if (b==1) hterm += irlm.U*op(sites,"N",b)*op(sites,"N",b+1);

            auto g = BondGate(sites,b,b+1,BondGate::tReal,dt/2.,hterm);
            gates.push_back(g);
        }
        //Create the gates exp(-i*tstep/2*hterm) in reverse
        for(int b = nTB-1; b >= 1; --b)
        {
            auto hterm = Kip(b-1,b)*op(sites,"Adag",b)*op(sites,"A",b+1);
            hterm += Kip(b,b-1)*op(sites,"A",b)*op(sites,"Adag",b+1);
            hterm += Kip(b-1,b-1)*op(sites,"N",b)*op(sites,"Id",b+1);
            hterm += Kip(b,b)*op(sites,"Id",b)*op(sites,"N",b+1);
            if (b==1) hterm += irlm.U*op(sites,"N",b)*op(sites,"N",b+1);

            auto g = BondGate(sites,b,b+1,BondGate::tReal,dt/2.,hterm);
            gates.push_back(g);
        }
        return gates;
    }

};


#endif // IRLM_H
