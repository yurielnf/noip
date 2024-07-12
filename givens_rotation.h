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
    T c=1, s=0;  ///< cos, sin, radius

    //GivensRot(size_t b_) : b(b_) {}

    /// build the J s.t.  J * (p,q)=(0,r) is go_right=true. Adapted from eigen.tuxfamily.org
    static GivensRot<T> createFromPair(size_t b, T p,  T q, bool go_right, T* r=nullptr);

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

    /// assuming that |z|=1 ??
    GivensRot<cmpx> operator*(cmpx z) const { GivensRot<cmpx> g{.b=b}; g.c=c*z; g.s=s*z; return g;}

    //void transposeInPlace() {s=-s;}
};


template<>
GivensRot<double> GivensRot<double>::createFromPair(size_t b, double p,  double q, bool go_right, double *r)
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
        if (r) *r = abs(p);
    }
    else if(p==Scalar(0))
    {
        g.c = Scalar(0);
        g.s = q<Scalar(0) ? Scalar(1) : Scalar(-1);
        if (r) *r = abs(q);
    }
    else if(abs(p) > abs(q))
    {
        Scalar t = q/p;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(p<Scalar(0))
            u = -u;
        g.c = Scalar(1)/u;
        g.s = -t * g.c;
        if (r) *r = p * u;
    }
    else
    {
        Scalar t = p/q;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(q<Scalar(0))
            u = -u;
        g.s = -Scalar(1)/u;
        g.c = -t * g.s;
        if (r) *r = q * u;
    }
    if (go_right) { g.s=-g.s; }
    return g;
}

template<>
GivensRot<double>::matrix22 GivensRot<double>::matrix() const { return {{c, -s},{s, c}}; }

template<>
GivensRot<cmpx> GivensRot<cmpx>::createFromPair(size_t b, cmpx p, cmpx q, bool go_right, cmpx *r)
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
        if (r) *r = g.c * p;
    }
    else if(p==Scalar(0))
    {
        g.c = 0;
        g.s = -q/abs(q);
        if (r) *r = abs(q);
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
            if (r) *r = p * u;
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
            if (r) *r = ps * u;
        }
    }
    if (go_right) { g.s=-conj(g.s); } // assuming g.c is real!!

    return g;
}

template<>
GivensRot<cmpx>::matrix22 GivensRot<cmpx>::matrix() const { return {{std::conj(c), -std::conj(s)},{s, c}}; }


template<class T>
arma::Mat<T> matrot_from_Givens(std::vector<GivensRot<T>> const& gates, size_t n)
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


//------------------------- set of Givens rotations -----------------------------------------

/// return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
/// the first column of the rotation will go to pos0,
/// the second to pos0+1, and so on.
template<class T>
static std::vector<GivensRot<T>> GivensRotForRot_left(arma::Mat<T> rot)
{
    //if (pos0+rot.n_cols>rot.n_rows) throw std::invalid_argument("GivensRotForRot: position outside the matrix");
    using namespace arma;
    std::vector<GivensRot<T>> givens;
    for(int j=0u; j<rot.n_cols; j++) {
        arma::Col<T> v=rot.col(j);
        std::vector<GivensRot<T>> gs1;
        for(int i=v.size()-2; i>=j; i--)
        {
            auto g=GivensRot<T>::createFromPair(i,v[i],v[i+1], false, &v[i]);
            gs1.push_back(g);
        }
        // if constexpr (std::is_same_v<cmpx,T>)
        //         if (!gs1.empty())
        //             gs1.back()=gs1.back()*v[j]; // to remove the phase
        auto rot1=matrot_from_Givens(gs1, rot.n_rows);
        arma::Mat<T> rotn = rot1*rot; //.cols(j,rot.n_cols-1) could start from j+1
        rot=rotn;
        for(auto g : gs1) givens.push_back(g);


        // v.clean(1e-15).print("v");
        // rot.clean(1e-15).print("rot");
        // rot1.clean(1e-15).print("rot1");
    }
    //rot.clean(1e-15).print("rot after extracting the Givens rotations");
    return givens;
}

#endif // GIVENS_ROTATION_H