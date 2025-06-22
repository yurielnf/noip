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
    using namespace itensor;
    using namespace arma;
    using namespace nlohmann;

    json j;
    {
        std::ifstream in("param.json");
        if (!in) throw invalid_argument("Please provide the param.json file");
        j=json::parse(in);
    }

    auto model0=Irlm_gs {j.at("irlm_gs")};

    auto ev1=arma::eig_sym(model0.irlm.kin_mat());
    auto ev2=arma::eig_sym(model0.irlm.star_kin());
    cout<<"kin-kin_star "<<arma::norm(ev1-ev2)<<endl;


    auto print_mps=[&]() {
        for(auto i=0; i<model0.psi.length(); i++)
            cout<<itensor::leftLinkIndex(model0.psi,i+1).dim()<<" ";
        cout << "\n";
    };

    print_mps();
    for(auto i=0;i<20;i++){
        model0.extractRepresentative();
        print_mps();
        model0.doDmrg();
        print_mps();
        model0.rotateToNaturalOrbitals();
        print_mps();
        cout<<i<<" "<<model0.nActive<<" "<<model0.energy<<endl;
    }

    cout<<"\n\nNormal dmrg\n";
    computeGS(model0.sites,model0.fullHamiltonian(false), &model0.psi);

    return 0;
}
