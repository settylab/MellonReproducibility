#! /usr/bin/env python
#SBATCH --job-name=mellon_stability_res_leiden
#SBATCH --nodes=1
#SBATCH --time=0-04:00:00
#SBATCH --out=/fh/fast/setty_m/user/dotto/CellDensities/data/slumr_out/%x_job-%A_%a_%N.log
#SBATCH --cpus-per-task=4

import sys
import os
import pandas as pd
import scanpy as sc

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

BASE_OUT_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/benchmarks/subsamples/stability_res_leiden_noNy/'

def make_estimator(sub_id, ad, n_landmarks=5000):
    number, odd = divmod(sub_id, 2)
    if odd:
        number = -number
    factor = 1.1**number
    leiden_res = 100 * factor
    print(f"Doing Leiden with resulotion {leiden_res}")
    sc.pp.neighbors(ad, use_rep="DM_EigenVectors")
    sc.tl.leiden(ad, leiden_res)
    n_clust = len(ad.obs['leiden'].unique())
    print(f"Found {n_clust} clusters, making landmarks...")
    X = ad.obsm["DM_EigenVectors"]
    landmarks = pd.DataFrame(X).groupby(ad.obs["leiden"].values).mean().values
    est = mellon.DensityEstimator(landmarks=landmarks, rank=1., method='percent')
    return est
    
if __name__ == "__main__":
    main_stability(BASE_OUT_PATH, make_estimator)
    print('Finished sucessfully.')