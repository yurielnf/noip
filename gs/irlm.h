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
    bool use_saved_ham;
    char ham_file_name[20]="ham.bin";
    char sites_file_name[20]="sites.bin";

    itensor::Fermion sites;

    IRLM(string tFile,string PFile, double U_, bool use_saved_ham_=false)
        :U(U_), use_saved_ham(use_saved_ham_)
    {
        tmat.load(tFile);
        Pmat.load(PFile);

        if (use_saved_ham)
            itensor::readFromFile(sites_file_name,sites);
        else
            sites=itensor::Fermion(length());
    }

    int length() const { return tmat.n_rows; }

    itensor::MPO Ham() const
    {
        if (use_saved_ham) {
            itensor::MPO ham(sites);
            itensor::readFromFile(ham_file_name,ham);
            return ham;
        }
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
                        if (c==d || a==b) continue;
                        ampo += coeff,"Cdag",a,"Cdag",b,"C",c,"C",d;
                    }
        }
        auto ham=toMPO(ampo);
        itensor::writeToFile(sites_file_name,sites);
        itensor::writeToFile(ham_file_name,ham);
        return ham;
    }

    itensor::MPS psi0() const
    {
        auto state = itensor::InitState(sites,"Emp");
        for( auto n : itensor::range1(length()) )
            if( n%2==0 ) state.set(n,"Occ");
        return itensor::randomMPS(state);
    }

    arma::mat cicj(itensor::MPS& psi) const
    {
        arma::mat cc(length(),length());

        for(auto i : itensor::range1(length())) {
            psi.position(i);
            auto psidag = dag(psi);
            psidag.prime();
            auto li_1 = leftLinkIndex(psi,i);

            auto n_i = op(sites,"N",i);
            cc(i-1,i-1)=elt(dag(prime(psi(i),"Site"))*n_i*psi(i));

            auto Adag_i = op(sites,"Adag",i);
            auto Cij0 = prime(psi(i),li_1)*Adag_i*psidag(i);
            for(auto j : itensor::range1(i+1,length())) {
                auto Cij=Cij0;
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
