#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --output=output.txt
#SBATCH --error=error.txt



export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=24

#export CMAKE_PREFIX_PATH=/home/jeannin/code/armadilloSandbox/armadillo-install/usr/:/home/jeannin/code/gslSandbox/gsl-install/usr/local/:/home/yurielnf/.local/lib/python3.7/site-packages/pybind11
#export PATH=/home/jeannin/code/cmake-3.23.0-rc2/bin:$PATH
#export PATH=/home/jeannin/code/lapackSandbox/lapack-build/lib:$PATH
#export CPATH=/home/jeannin/code/gslSandbox/gsl-install/usr/local/include:/home/yurielnf/opt
#export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH

time /home/yurielnf/projects/it_irlm/tdvp/build/example/irlm_star
