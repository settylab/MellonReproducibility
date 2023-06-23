#! /usr/bin/env python
#SBATCH --job-name=mellon_stability_n_maxmin_noNy
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
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

BASE_OUT_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/benchmarks/subsamples/stability_n_maxmin_noNy/'

def make_estimator(sub_id, ad, n_landmarks=5000):
    number, odd = divmod(sub_id, 2)
    if odd:
        number = -number
    factor = 1.1**number
    new_n_landmarks = n_landmarks * factor
    X = ad.obsm["DM_EigenVectors"]
    ncomps = X.shape[1]
    if n_landmarks < ncomps:
        print(f'Number of clusers is {n_landmarks} which is lower then number of compnents {ncomps}. Stopping.')
        sys.exit(0)
    idx = palantir.core._max_min_sampling(pd.DataFrame(X), new_n_landmarks)
    est = mellon.DensityEstimator(landmarks=X[idx, :], rank=1., method='percent')
    return est
    
if __name__ == "__main__":
    main_stability(BASE_OUT_PATH, make_estimator)
    print('Finished sucessfully.')