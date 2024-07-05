#include<catch2/catch.hpp>
#include "givens_rotation.h"

using namespace arma;
using namespace std;

TEST_CASE("arma") {
    mat A = { {1, 3, 5},
              {2, 4, 6} };
    //A.print("A=");
}


TEST_CASE( "GivensRotation real" )
{
    double tol=1e-14;
    vec v(2, fill::randu);
    auto g=GivensRot<>::createFromPair(0, v[0], v[1], true);

    SECTION( "definition" )
    {
        vec y=g.matrix()*v;
        REQUIRE(std::abs(norm(y)/norm(v)-1)<tol);
        REQUIRE(std::abs(y[0]/y[1])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilogMatrix();
        REQUIRE(norm(g.matrix()-expmat(h*cmpx(0,-1)))<tol);
        REQUIRE(norm(h-h.t())<tol);
    }

    SECTION("3d case")
    {
        vector<GivensRot<>> gs;
        arma::vec v={0.5,1.5,-1}, vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<>::createFromPair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_Givens(gs, v.size());
        vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[0]/y[2])<tol);
        REQUIRE(std::abs(y[1]/y[2])<tol);
    }

    SECTION("matrix case")
    {
        vector<GivensRot<>> gs;
        arma::mat A(3,3, arma::fill::randu), evec;
        arma::vec eval;
        A = A*A.t();
        arma::eig_sym(eval,evec,A);
        arma::vec v=evec.col(0), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<>::createFromPair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_Givens(gs,A.n_cols);
        REQUIRE(norm(rot*rot.t()-eye(size(rot)))<tol);
        arma::mat Arot=rot*A*rot.t();
        REQUIRE(std::abs(Arot(2,2)/eval(0)-1)<tol*norm(A));
        REQUIRE(std::abs(Arot(0,2)/norm(A))<tol);
        REQUIRE(std::abs(Arot(1,2)/norm(A))<tol);
    }
}

TEST_CASE( "GivensRotation complex" )
{
    double tol=1e-14;
    cx_vec v(2, fill::randu);
    auto g=GivensRot<cmpx>::createFromPair(0, v[0], v[1], true);

    SECTION( "definition" )
    {
        cx_vec y=g.matrix()*v;
        REQUIRE(std::abs(arma::norm(y)/arma::norm(v)-1)<tol);
        REQUIRE(std::abs(y[0]/y[1])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilogMatrix();
        REQUIRE(norm(h-h.t())<tol);
        REQUIRE(norm(g.matrix()-expIH(h))<tol);
    }

    SECTION("3d case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_vec v(3,fill::randu), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::createFromPair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_Givens(gs, v.size());
        cx_vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[0]/y[2])<tol);
        REQUIRE(std::abs(y[1]/y[2])<tol);
    }

    SECTION("matrix case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_mat A(3,3, arma::fill::randu), evec;
        arma::vec eval;
        A = A*A.t();
        arma::eig_sym(eval,evec,A);
        arma::cx_vec v=evec.col(0), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::createFromPair(b,v[i],v[i+1],true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_Givens(gs,A.n_cols);
        REQUIRE(norm(rot*rot.t()-eye(size(rot)))<tol);
        REQUIRE(norm(rot.t()*rot-eye(size(rot)))<tol);
        arma::cx_mat Arot=rot*A*rot.t();
        REQUIRE(std::abs(Arot(2,2)/eval(0)-1.0)<tol*norm(A));
        REQUIRE(std::abs(Arot(0,2)/norm(A))<tol);
        REQUIRE(std::abs(Arot(1,2)/norm(A))<tol);
    }
}


TEST_CASE( "GivensRotation complex left" )
{
    double tol=1e-14;
    cx_vec v(2, fill::randu);
    auto g=GivensRot<cmpx>::createFromPair(0, v[0], v[1], false);

    SECTION( "definition" )
    {
        cx_vec y=g.matrix()*v;
        REQUIRE(std::abs(arma::norm(y)/arma::norm(v)-1)<tol);
        REQUIRE(std::abs(y[1]/y[0])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilogMatrix();
        REQUIRE(norm(h-h.t())<tol);
        REQUIRE(norm(g.matrix()-expIH(h))<tol);
    }

    SECTION("3d case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_vec v(3,fill::randu), vc=v;
        for(int i=v.size()-2; i>=0; i--)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::createFromPair(b,v[i],v[i+1], false, &v[i]);
            gs.push_back(g);
        }
        auto rot=matrot_from_Givens(gs, v.size());
        cx_vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[2]/y[0])<tol);
        REQUIRE(std::abs(y[1]/y[0])<tol);
    }
}

TEST_CASE("set of Givens")
{
    SECTION("basic")
    {
        arma::cx_mat X(5,5, fill::randu), U, V;
        vec s;
        svd(U,s,V,X);
        //s.print("s");
        //V.print("V");
        // auto givens=GivensRotForRot_right(V,V.n_cols-1);
        auto givens=GivensRotForRot_left(V.head_cols(5).eval());
        auto G=matrot_from_Givens(givens,V.n_rows).t().eval();
        //G.print("givens");
        //cout<<norm(G.t()*G-eye<decltype(X)>(size(V)))<<endl;
    }

    SECTION("kin")
    {
        int len=5;
        arma::cx_mat kin= cx_mat(3,2,fill::randu)*
                cx_mat(2,len, fill::randu), U, V;
        vec s;
        svd_econ(U,s,V,kin);
        auto givens=GivensRotForRot_left(V.cols(0,1).eval());
        auto rot1=matrot_from_Givens(givens,V.n_rows);
        rot1.print("rot1");
        kin.print("kin");
        (kin*rot1.t()).eval().clean(1e-15).print("kin after rot f");
    }

}
