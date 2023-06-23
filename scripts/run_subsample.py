#! /usr/bin/env python
#SBATCH --job-name=mellon_benchmarks
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --out=/fh/fast/setty_m/user/dotto/CellDensities/data/slumr_out/%x_job-%A_%a_%N.log
#SBATCH --cpus-per-task=36
#SBATCH --exclusive

import sys
import os

os.environ["JAX_PLATFORM_NAME"]="cpu"

sys.path = [os.getcwd()] + sys.path
from benchmark_utils import *

BASE_OUT_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/benchmarks/subsamples/hierarchical/'

def run_density_estimation(ad, X, n_landmarks=5000):
    
    estimator = mellon.DensityEstimator(n_landmarks=n_landmarks)
    print('density...')
    estimator._set_x(X)
    estimator._prepare_attribute("nn_distances")
    estimator._prepare_attribute("d")
    estimator._prepare_attribute("mu")
    estimator._prepare_attribute("ls")
    estimator._prepare_attribute("cov_func")
    estimator._prepare_attribute("landmarks")
    estimator._prepare_attribute("L")
    estimator._prepare_attribute("initial_value")
    estimator._prepare_attribute("transform")
    estimator._prepare_attribute("loss_func")
    estimator.run_inference()
    estimator.process_inference(build_predict=False)
    estimator._set_log_density_func()
    ad.obs['log_density'] = estimator.predict(X)
    save_estimator_stats(ad, estimator)
        
    
if __name__ == "__main__":
    main(BASE_OUT_PATH, run_density_estimation)
    print('Finished sucessfully.')