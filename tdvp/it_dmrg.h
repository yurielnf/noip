#ifndef IT_DMRG_H
#define IT_DMRG_H

#include "fermionic.h"

struct it_dmrg {
    int bond_dim=64;
    int nIter_diag=4;
    double rho_cutoff=1e-10;
    double noise=1e-8;

    int nsweep=0;
    double energy=0;
    HamSys hamsys;
    itensor::MPS psi;

    it_dmrg(HamSys const& hamsys_)
        : hamsys {hamsys_}
        , psi {itensor::randomMPS(hamsys_.sites)}
    {}

    void iterate()
    {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = bond_dim;
        sweeps.cutoff() = rho_cutoff;
        sweeps.niter() = nIter_diag;
        sweeps.noise() = noise;
        energy=itensor::dmrg(psi,hamsys.ham,sweeps, {"Quiet",true});
        nsweep++;
    }
};

#endif // IT_DMRG_H
