#! /bin/bash

#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --constraint="type_a"
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

mkdir -p bemchmark_results

module load module load nvidia_sdk/nvhpc/23.5
N_RUNS=10
SEED=12345

echo starting
echo n_runs: $N_RUNS seed: $SEED
echo
for matrix_size in 2000 3000 4000 5000; do
    echo matrix_size: $matrix_size
    ./a.out $matrix_size $N_RUNS $SEED >bemchmark_results/matrix_size_${matrix_size}.csv
done
