#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 30:00
#SBATCH -J hpgmg

export GASNET_MAX_SEGSIZE='4096MB'
export GASNET_PHYSMEM_MAX=16G
export GASNET_BACKTRACE=1
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.json
# export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.no_interconnect.json
export OMP_NUM_THREADS=12
export HCLIB_WORKERS=$OMP_NUM_THREADS

# cd $HOME/hclib/test/cpp
# srun -N 1 -n 1 ./test_all.sh

cd $HOME/hpgmg
LD_LIBRARY_PATH=$HCLIB_HOME/modules/mpi/lib:$LD_LIBRARY_PATH srun -N 2 -n 4 --ntasks-per-node=2 ./build/bin/hpgmg-fv 7 8
