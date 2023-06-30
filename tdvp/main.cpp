#include "irlm.h"
#include "it_dmrg.h"
#include "it_tdvp.h"

#include<iostream>

using namespace std;

int main()
{
    HamSys sys=IRLM {.L=20, .t=0.5, .V=0.15, .U=0.1}.HamStar();

    cout<<"bond dimensions of H:\n";
    for(int i=1; i<sys.sites.length(); i++)
        cout<<rightLinkIndex(sys.ham,i).dim()<<" ";
    cout<<endl;

    // solve the gs of system
    it_dmrg sol_gs {sys};
    for(auto i=0u; i<5; i++) {
        sol_gs.iterate();
    }

    // evolve the psi with new Hamiltonian
    sys=IRLM {.L=20, .t=0.5, .V=0.15, .U=-0.5}.HamStar();
    auto psi=sol_gs.psi.replaceSiteInds(sys.sites.inds());
    it_tdvp sol {sys, psi};
    for(auto i=0u; i<10; i++) {
        sol.iterate();
    }

    return 0;
}
