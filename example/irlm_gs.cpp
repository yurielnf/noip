#include "irlm_gs.h"
#include "it_dmrg.h"
#include "it_tdvp.h"
#include "irlm_gs.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include<itensor/all.h>
#include <nlohmann/json.hpp>

using namespace std;

struct State {
    itensor::MPS psi;
    itensor::Fermion sites;
};

auto computeGS(itensor::Fermion const& sites, itensor::MPO const& ham, itensor::MPS *psi=nullptr)
{
    cout<<"bond dimension of H: "<< maxLinkDim(ham) << endl;
    it_dmrg sol_gs = psi ? it_dmrg {HamSys{.sites=sites,.ham=ham}, *psi} :
                           it_dmrg {HamSys{.sites=sites,.ham=ham}};
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-3;
    sol_gs.silent=true;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<32; i++) {
        if (i==10) { sol_gs.noise=1e-7; sol_gs.bond_dim=256; }
        else if (i==26) { sol_gs.noise=1e-9; sol_gs.bond_dim=512; }
        else if (i==28) sol_gs.noise=0;
        if (false && i==11) {
            // Global subspace expansion
//            std::vector<double> epsilonK(3,1E-8);
            std::vector<int> maxDimK(3,0.5*itensor::maxLinkDim(sol_gs.psi));

            itensor::addBasis(sol_gs.psi,sol_gs.hamsys.ham,maxDimK,
                     {"Cutoff", 1e-9,
                      "Method", "DensityMatrix",
                      "KrylovOrd",3,
                      "DoNormalize", true,
                      "Quiet",true,
                      "Silent",true});
        }
        sol_gs.iterate();
        //sol_gs.psi.orthogonalize({"Cutoff",1e-9});
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return sol_gs;
}


auto prepareSlater(itensor::Fermion const& sites, arma::vec ek, int nPart)
{
    auto state = itensor::InitState(sites,"0");
    arma::uvec iek=arma::sort_index(ek);
    double energy=0;
    for(int j = 0; j < nPart; j++) {
        state.set(iek[j]+1,"1");
        energy += ek[iek[j]];
    }
    std::cout << " Slater energy: " << energy << std::endl;
    return itensor::MPS(state);
}




NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IrlmData,L,t,V,U,ed,connected)

int main()
{    
    using namespace nlohmann;

    json j;
    {
        std::ifstream in("param.json");
        if (!in) throw invalid_argument("Please provide the param.json file");
        j=json::parse(in);
    }

    itensor::cpu_time t0;
    t0.mark();
    auto model0=Irlm_gs {j.at("irlm_gs")};
    cout<<"initialization: "<<t0.sincemark().wall<<endl;
    t0.mark();

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    for(auto i=0;i<30;i++){
        model0.iterate();
        cout<<i+1<<" "<<model0.nActive<<" "<<model0.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }

    // cout<<"\n\nNormal dmrg\n";
    //auto sol_gs=computeGS(model0.sites,model0.fullHamiltonian(true));

    return 0;
}
