#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 512
#SBATCH -t 20:00
#SBATCH -J hpgmg
#SBATCH --exclusive
#SBATCH --exclusive
#SBATCH --mail-type=ALL

export GASNET_MAX_SEGSIZE='4096MB'
export GASNET_PHYSMEM_MAX=16G
export GASNET_BACKTRACE=1
export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.json
# export HCLIB_LOCALITY_FILE=$HCLIB_HOME/locality_graphs/edison.no_interconnect.json
export OMP_NUM_THREADS=12
export HCLIB_WORKERS=$OMP_NUM_THREADS

# cd $HOME/hclib/test/cpp
# srun -N 1 -n 1 ./test_all.sh

NRUNS=10
NPROCS=1024

for RUN in $(seq $NRUNS); do
    echo '========== HIPER =========='
    cd $HOME/hpgmg
    LD_LIBRARY_PATH=$HCLIB_HOME/modules/mpi/lib:$LD_LIBRARY_PATH srun --ntasks=$NPROCS --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 --cpu_bind=verbose,cores ./build/bin/hpgmg-fv 7 8
done

for RUN in $(seq $NRUNS); do
    echo '========== VANILLA =========='
    cd $HOME/hpgmg-vanilla
    LD_LIBRARY_PATH=$HCLIB_HOME/modules/mpi/lib:$LD_LIBRARY_PATH srun --ntasks=$NPROCS --ntasks-per-node=2 --ntasks-per-socket=1 --cpus-per-task=12 --cpu_bind=verbose,cores ./build/bin/hpgmg-fv 7 8
done
