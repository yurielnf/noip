#include "irlm.h"

using namespace itensor;

int 
main(int argc, char* argv[])
{
    if (argc!=4) { cout<<"usage: <tFileName> <PFileName> <U>"; return 0; }
    auto sweeps = Sweeps(10); //number of sweeps is 5
    sweeps.maxdim() = 10,20,100,100,200,300;
    sweeps.cutoff() = 1E-14;
    sweeps.niter() = 2,2,2,2,20;
    auto sol=IRLM(argv[1], argv[2], atof(argv[3]));
    auto [energy,psi] = dmrg(sol.Ham(),sol.psi0(),sweeps,"Quiet");

    sol.cicj(psi).print("cicj=");

    return 0;
}


