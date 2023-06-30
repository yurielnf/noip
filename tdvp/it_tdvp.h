#ifndef IT_TDVP_H
#define IT_TDVP_H

#include "fermionic.h"
#include "tdvp.h"
#include "basisextension.h"

struct it_tdvp {
    int bond_dim=64;
    int nIter_diag=10;
    double rho_cutoff=1e-12;
    double noise=1e-8;
    std::complex<double> dt={0, 0.1};
    bool do_normalize=false;
    double err_goal=1e-7;

    int nsweep=0;
    double energy=0;
    HamSys hamsys;
    itensor::MPS psi;

    it_tdvp(HamSys const& hamsys_, itensor::MPS const& psi_) : hamsys(hamsys_), psi(psi_) {}

    std::complex<double> time() const { return dt * static_cast<double>(nsweep); }

    void iterate()
    {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = bond_dim;
        sweeps.cutoff() = rho_cutoff;
        sweeps.niter() = nIter_diag;

        if (noise != 0)
        {
            // Global subspace expansion
            std::vector<double> epsilonK = {1E-12, 1E-12};
            addBasis(psi,hamsys.ham,epsilonK,
                     {"Cutoff",noise,
                      "Method","DensityMatrix",
                      "KrylovOrd",3,
                      "DoNormalize", do_normalize,
                      "Quiet",true});
        }

        // TDVP sweep
        energy = tdvp(psi,hamsys.ham,-dt,sweeps,
                      {"Truncate",true,
                       "DoNormalize", do_normalize,
                       "Quiet",true,
                       "NumCenter",1,
                       "ErrGoal", err_goal});


        nsweep++;
    }
};

#endif // IT_TDVP_H
