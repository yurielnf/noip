#include <iostream>
#include <iomanip>
#include<itensor/all.h>
#include <nlohmann/json.hpp>

#include "iscboson.h"

using namespace std;


int main(int argc, char **argv)
{
    using namespace itensor;
    using namespace nlohmann;

    cout<<"Hola mundo\n"<<setprecision(14);

    int len = 10;
    HamBSys sys=Iscb(IscbParam{.L=len, .wp=1, .alpha=0.0, .eJ=100}).Ham();


    auto sweeps = Sweeps(10);
    sweeps.maxdim() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-10;
    sweeps.noise()=1e-3,1e-3,1e-3,1e-7,1e-9;

    auto psi0 = randomMPS(sys.sites);

    auto [energy,psi] = dmrg(sys.ham,psi0,sweeps, {"Quiet", true, "Silent", false});

    println("Ground State Energy = ",energy);

    //
    // Measuring ni
    //

    println("\n <(Adag+A)^2> ");
    for( auto j : range1(len) )
    {
        //re-gauge psi to get ready to measure at position j
        psi.position(j);

        auto ket = psi(j);
        auto bra = dag(prime(ket,"Site"));

        auto Njop = op(sys.sites,"A*A",j);
        Njop += op(sys.sites,"Adag*Adag",j);
        Njop += op(sys.sites,"A*Adag",j);
        Njop += op(sys.sites,"Adag*A",j);

        //take an inner product
        auto nj = elt(bra*Njop*ket);
        printfln("%.12f",nj);
    }



    return 0;
}
