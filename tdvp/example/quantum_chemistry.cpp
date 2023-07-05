#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include<iostream>

using namespace std;
using namespace itensor;
using namespace arma;

void FreeFermion(arma::mat const& K, AutoMPO& H)
{
    int L=K.n_rows;
    for(auto i=0; i<L; i++)
        for(auto j=0; j<L; j++)
            if (fabs(K(i,j))>1e-14)
                H += K(i,j),"Cdag",i+1,"C",j+1;
}

void InteractingFermion(arma::mat const& Vijkl, AutoMPO& H)
{
    int L=sqrt(Vijkl.n_rows);
    for(auto i=0; i<L; i++)
        for(auto j=i+1; j<L; j++)
            for(auto k=0; k<L; k++)
                for(auto l=k+1; l<L; l++)
                    if (fabs(Vijkl(i+j*L,k+l*L))>1e-14)
                        H += Vijkl(i+j*L,k+l*L), "Cdag",i+1, "Cdag", j+1, "C", k+1, "C", l+1;
}



void testMPO()
{
    cout<<"\n------- Chemistry Ham versus L ------\n";
    int L=30;
    mat K3(L,L,arma::fill::randu);
    K3=K3*K3.t();
    mat V(L*L,L*L,arma::fill::randu);
     cout<<"L nterms D theoretical_D\n";
    for(auto len=6; len<=L; len+=4) {
        auto Kin=K3.submat(0,0,len-1,len-1);
        auto Vijkl=V.submat(0,0,len*len-1,len*len-1);
        auto sites=Fermion(len, {"ConserveQNs=",true});
        AutoMPO H(sites);
        FreeFermion(Kin,H);
        InteractingFermion(Vijkl, H);
        auto mpo=toMPO(H);
        cout<<len<<" "<<H.size()<<" "<< maxLinkDim(mpo) <<" "<< 2*pow(len/2,2)+3*(len/2)+2 << endl;
    }

}


int main()
{
    testMPO();

    return 0;
}
