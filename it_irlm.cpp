#include "irlm.h"

using namespace itensor;

int 
main(int argc, char* argv[])
{
    if (argc!=4) { cout<<"usage: <tFileName> <PFileName> <U>"; return 0; }
    auto sweeps = Sweeps(15); //number of sweeps is 5
    sweeps.maxdim() = 10,20,50,50,60;
    sweeps.cutoff() = 1E-12;
    sweeps.niter() = 2,2,2,2,2,10,10,20,20,30,50,50,100,100,200;
//    sweeps.noise() = 1e-1,1e-2,1e-3,1e-5,1e-7,1e-8,1e-9;
    auto sol=IRLM(argv[1], argv[2], atof(argv[3]), true);
    auto [energy,psi] = dmrg(sol.Ham(),sol.psi0(),sweeps,"Quiet");

    auto cc=sol.cicj(psi);
    cc.save("cicj.txt",arma::raw_ascii);
    arma::eig_sym(cc).save("eval.txt",arma::raw_ascii);
    cc.diag().eval().save("ni.txt",arma::raw_ascii);

    cout<<"bond dimensions:\n";
    for(int i=1; i<sol.length(); i++)
        cout<<rightLinkIndex(psi,i).dim()<<" ";
    cout<<endl;

    return 0;
}


