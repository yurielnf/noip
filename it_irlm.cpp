#include "itensor/all.h"

using namespace itensor;

int 
main()
{
    int N = 20;
    auto sites = Fermion(N);
    auto t = 0.5;
    auto ampo = AutoMPO(sites);
    for( auto n : range1(N-1) )
        {
        ampo += -t,"Cdag",n,"C",n+1;
        ampo += -t,"Cdag",n+1,"C",n;
        }
    auto H = toMPO(ampo);
    auto state = InitState(sites,"Emp");
    for( auto n : range1(N) ) if( n%2==0 ) state.set(n,"Occ");
    auto sweeps = Sweeps(5); //number of sweeps is 5
    sweeps.maxdim() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 2,2,2,2,10;
    auto [energy,psi] = dmrg(H,randomMPS(state),sweeps,"Quiet");

    auto i = 5;
    auto j = 10;

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

    auto result = elt(Cij); //or eltC(Cij) if expecting complex

    std::cout<<"<ci cj>="<<result << "\n";

    return 0;
}


