#ifndef GIVENS_ROTATION_H
#define GIVENS_ROTATION_H


#include<armadillo>
#include <itensor/all.h>

using cmpx=std::complex<double>;

/// compute the exp(-i H) assuming H is Hermitian
template<class T>
arma::cx_mat expIH(arma::Mat<T> const& H)
{
    arma::Mat<T> evec;
    arma::vec eval;
    arma::eig_sym(eval,evec,H);
    return evec * arma::diagmat(arma::exp(eval*cmpx(0,-1))) * evec.t();
}


template<class T>
std::pair<arma::cx_vec,arma::cx_mat> eig_unitary(const arma::Mat<T>& A)
{
    using namespace arma;
    cx_vec eval, eval2;
    cx_mat evec, Q, R;
    eig_gen(eval, evec, A);
    qr(Q, R, evec);
    cx_mat RDR=R*diagmat(eval)*R.i();
    eval2=RDR.diag();

#ifndef NDEBUG
        double err=norm(A-Q*diagmat(eval2)*Q.t());
        std::cout<<"error diag RDR="<<norm(RDR-diagmat(eval))<<std::endl;
        std::cout<<"err eig_unitary="<<err<<std::endl;
        std::cout<<"err |eval|-1="<<norm(arma::abs(eval2)-ones(A.n_cols))<<std::endl;
#endif

    return {eval2,Q};
}

template<class T=double>
struct GivensRot {
    using matrix22=typename arma::Mat<T>::template fixed<2,2>;

    size_t b;     ///< bond b --- b+1
    T c=1, s=0, r=1;  ///< cos, sin, radius

    //GivensRot(size_t b_) : b(b_) {}

    /// build the J s.t.  J * (p,q)=(0,r) is go_right=true. Adapted from eigen.tuxfamily.org
    static GivensRot<T> createFromPair(size_t b, T p,  T q, bool go_right);

    //double angle() const { return atan2(s,c); }

    matrix22 matrix() const;

    /// the underline "Hamiltonian" the output is Hermitian.
    arma::cx_mat ilogMatrix() const
    {
        matrix22 rot=matrix();
        auto [eval,evec]=eig_unitary(rot);
        arma::vec eval2=arma::real( arma::log(eval)*cmpx(0,1) );
        return evec * arma::diagmat(eval2) * evec.t();
    }

    //void transposeInPlace() {s=-s;}
};


template<>
GivensRot<double> GivensRot<double>::createFromPair(size_t b, double p,  double q, bool go_right)
{
    using Scalar=double;
    using std::sqrt;
    using std::abs;

    GivensRot<double> g {.b=b};
    if (go_right) std::swap(p,q); // to eliminate the p instead of q.
    if(q==Scalar(0))
    {
        g.c = p<Scalar(0) ? Scalar(-1) : Scalar(1);
        g.s = Scalar(0);
        g.r = abs(p);
    }
    else if(p==Scalar(0))
    {
        g.c = Scalar(0);
        g.s = q<Scalar(0) ? Scalar(1) : Scalar(-1);
        g.r = abs(q);
    }
    else if(abs(p) > abs(q))
    {
        Scalar t = q/p;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(p<Scalar(0))
            u = -u;
        g.c = Scalar(1)/u;
        g.s = -t * g.c;
        g.r = p * u;
    }
    else
    {
        Scalar t = p/q;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(q<Scalar(0))
            u = -u;
        g.s = -Scalar(1)/u;
        g.c = -t * g.s;
        g.r = q * u;
    }
    if (go_right) { g.s=-g.s; }
    return g;
}

template<>
GivensRot<double>::matrix22 GivensRot<double>::matrix() const { return {{c, -s},{s, c}}; }

template<>
GivensRot<cmpx> GivensRot<cmpx>::createFromPair(size_t b, cmpx p, cmpx q, bool go_right)
{
    using Scalar=cmpx;
    using RealScalar=double;
    using std::sqrt;
    using std::abs;
    using std::conj;

    GivensRot<cmpx> g {.b=b};
    if (go_right) std::swap(p,q); // to eliminate the p instead of q.
    if(q==Scalar(0))
    {
        g.c = std::real(p)<0 ? Scalar(-1) : Scalar(1);
        g.s = 0;
        g.r = g.c * p;
    }
    else if(p==Scalar(0))
    {
        g.c = 0;
        g.s = -q/abs(q);
        g.r = abs(q);
    }
    else
    {
        RealScalar p1 = std::abs(p);
        RealScalar q1 = std::abs(q);
        if(p1>=q1)
        {
            Scalar ps = p / p1;
            RealScalar p2 = std::norm(ps);
            Scalar qs = q / p1;
            RealScalar q2 = std::norm(qs);

            RealScalar u = sqrt(RealScalar(1) + q2/p2);
            if(std::real(p)<RealScalar(0))
                u = -u;

            g.c = Scalar(1)/u;
            g.s = -qs*conj(ps)*(g.c/p2);
            g.r = p * u;
        }
        else
        {
            Scalar ps = p / q1;
            RealScalar p2 = std::norm(ps);
            Scalar qs = q / q1;
            RealScalar q2 = std::norm(qs);

            RealScalar u = q1 * sqrt(p2 + q2);
            if(std::real(p)<RealScalar(0))
                u = -u;

            p1 = abs(p);
            ps = p/p1;
            g.c = p1/u;
            g.s = -conj(ps) * (q/u);
            g.r = ps * u;
        }
    }
    if (go_right) { g.s=-conj(g.s); } // assuming g.c is real!!

    return g;
}

template<>
GivensRot<cmpx>::matrix22 GivensRot<cmpx>::matrix() const { return {{std::conj(c), -std::conj(s)},{s, c}}; }


template<class T>
arma::Mat<T> matrot_from_Givens(std::vector<GivensRot<T>> const& gates, size_t n=0)
{
    if (n==0) {
        for(const GivensRot<T>& g : gates) if (g.b>n) n=g.b;
        n+=2;
    }
    arma::Mat<T> rot(n,n, arma::fill::eye);
    for(int i=gates.size()-1; i>=0; i--) { // apply to the right in reverse
//    for(auto i=0u; i<gates.size(); i++) {
        const GivensRot<T>& g=gates[i];
        rot.submat(0,g.b,n-1,g.b+1) = rot.submat(0,g.b,n-1,g.b+1) * g.matrix();
    }
    return rot;
}


#endif // GIVENS_ROTATION_H
