#ifndef IRLM_H
#define IRLM_H

#include<armadillo>
#include <itensor/all.h>

using namespace std;

class IRLM
{

public:
    arma::mat tmat, Pmat;
    double U;
    double tol=1e-14;

    itensor::Fermion sites;

    IRLM(string tFile,string PFile, double U_)
        :U(U_)
    {
        tmat.load(tFile);
        Pmat.load(PFile);

        cout<<"P*P.t()-1 = "<< arma::norm(Pmat*Pmat.t()-arma::mat(length(),length(), arma::fill::eye)) << endl;

        sites=itensor::Fermion(length());
    }

    int length() const { return tmat.n_rows; }

    itensor::MPO Ham() const
    {
        auto L=length();
        auto ampo = itensor::AutoMPO(sites);

        // kinetic energy bath
        for(auto i : itensor::range1(L))
            for(auto j : itensor::range1(L))
        {
            if (fabs(tmat(i-1,j-1))<tol) continue;
            ampo += tmat(i-1,j-1),"Cdag",i,"C",j;
        }

        //interaction
        for(auto a : itensor::range1(L)) { cout<<a<<" "; cout.flush();
            for(auto b : itensor::range1(L))
                for(auto c : itensor::range1(L))
                    for(auto d : itensor::range1(L)) {
                        double coeff=U*Pmat(0,a-1)*Pmat(1,b-1)*Pmat(1,c-1)*Pmat(0,d-1);
                        if (c==d || a==b || fabs(coeff)<tol) continue;
                        ampo += coeff,"Cdag",a,"Cdag",b,"C",c,"C",d;
                    }
        }
        return toMPO(ampo);
    }

    itensor::MPS psi0() const
    {
        auto state = itensor::InitState(sites,"Emp");
        for( auto n : itensor::range1(length()) )
            if( n%2==0 ) state.set(n,"Occ");
        return itensor::randomMPS(state);
    }

    double cicj(itensor::MPS& psi,int i,int j)
    {
        auto Adag_i = op(sites,"Adag",i);
        auto A_j = op(sites,"A",j);

        //'gauge' the MPS to site i
        //any 'position' between i and j, inclusive, would work here
        psi.position(i);

        auto psidag = dag(psi);
        psidag.prime();

        //index linking i to i-1:
        auto li_1 = leftLinkIndex(psi,i);
        auto Cij = prime(psi(i),li_1)*Adag_i*psidag(i);
        for(int k = i+1; k < j; ++k)
            {
            Cij *= psi(k);
            Cij *= op(sites,"F",k); //Jordan-Wigner string
            Cij *= psidag(k);
            }
        //index linking j to j+1:
        auto lj = rightLinkIndex(psi,j);
        Cij *= prime(psi(j),lj);
        Cij *= A_j;
        Cij *= psidag(j);

        return elt(Cij); //or eltC(Cij) if expecting complex
    }

    arma::mat cicj(itensor::MPS& psi) const
    {
        arma::mat cc(length(),length());
        for(auto i : itensor::range1(length())) {
            auto Adag_i = op(sites,"Adag",i);
            psi.position(i);
            auto psidag = dag(psi);
            psidag.prime();
            //index linking i to i-1:
            auto li_1 = leftLinkIndex(psi,i);
            auto Cij = prime(psi(i),li_1)*Adag_i*psidag(i);
            for(auto j : itensor::range1(i+1,length())) {
                auto A_j = op(sites,"A",j);
                for(int k = i+1; k < j; ++k)
                {
                    Cij *= psi(k);
                    Cij *= op(sites,"F",k); //Jordan-Wigner string
                    Cij *= psidag(k);
                }
                //index linking j to j+1:
                auto lj = rightLinkIndex(psi,j);
                Cij *= prime(psi(j),lj);
                Cij *= A_j;
                Cij *= psidag(j);

                cc(i-1,j-1)=cc(j-1,i-1)=elt(Cij);
            }
        }
        return cc;
    }
};


#endif // IRLM_H
