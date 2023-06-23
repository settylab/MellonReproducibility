#! /usr/bin/env python
#SBATCH --job-name=mellon_stability_n_landmarks
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --out=/fh/fast/setty_m/user/dotto/CellDensities/data/slumr_out/%x_job-%A_%a_%N.log
#SBATCH --cpus-per-task=4

import sys
import os

cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if cpus:
    os.environ['OPENBLAS_NUM_THREADS'] = cpus
    os.environ['MKL_NUM_THREADS'] = cpus
    os.environ["NUM_INTER_THREADS"]=cpus
    os.environ["NUM_INTRA_THREADS"]=cpus
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=true "
                           f"intra_op_parallelism_threads={cpus}")

sys.path = [os.getcwd()] + sys.path
from benchmark_utils import *

BASE_OUT_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/benchmarks/subsamples/stability_n_landmarks/'

def make_estimator(sub_id, X, n_landmarks=5000):
    number, odd = divmod(sub_id, 2)
    if odd:
        number = -number
    marks = 1.1**number
    new_n_landmarks = n_landmarks * factor
    if n_landmarks <= 1:
        print(f'Factor is {factor} and base n_landmarks is {n_landmarks} resulting {new_n_landmarks} <= 1. Stopping.')
        sys.exit(0)
    est = mellon.DensityEstimator(n_landmarks=int(new_n_landmarks))
    return est
    
if __name__ == "__main__":
    main_stability(BASE_OUT_PATH, make_estimator)
    print('Finished sucessfully.')