#include "irlm.h"

using namespace itensor;

int 
main()
{
    auto sweeps = Sweeps(5); //number of sweeps is 5
    sweeps.maxdim() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 2,2,2,2,10;
    auto sys=IRLM("","",-0.5);
    auto [energy,psi] = dmrg(sys.Ham(),sys.psi0(),sweeps,"Quiet");

    std::cout<<"<ci cj>="<<sys.cicj(psi,5,10) << "\n";

    return 0;
}


