#!/bin/bash

# Usage: ./run_job.sh <num_cores> <num_threads> <params>

num_cores=${1:-4}
num_threads=${2:-16}
params=${3:-default_param}

bsub <<EOF
#BSUB -n $num_cores
#BSUB -W 00:15
#BSUB -o "omp_job.%J.out"
#BSUB -e "omp_job.%J.err"
OMP_NUM_THREADS=$num_threads ./out $params
EOF

