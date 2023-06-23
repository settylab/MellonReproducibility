import sys
import os
from pathlib import Path

import cProfile
import pstats
import io
from memory_profiler import profile

import pandas as pd
from anndata import read_h5ad
import mellon
import palantir
import scanpy as sc

BASE_IN_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/subsamples/hierarchical/'
#BASE_OUT_PATH = '/fh/fast/setty_m/user/dotto/CellDensities/data/benchmarks/subsamples/hierarchical_noNy/'

def get_args():
    print(sys.argv)
    dataset = sys.argv[1]
    sub_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if len(sys.argv)>2:
        n_landmarks = int(sys.argv[2])
    else:
        n_landmarks = 5000
        
    return dataset, sub_id, n_landmarks
    

def get_digit_count(data_path):
    for fl in os.listdir(data_path):
        if fl.startswith('sub_') and fl.endswith('.h5ad'):
            break
    else:
        raise ValueError(
            'Could not fine any valid file names '
            '(starting with "sub_" and ending with ".h5ad") '
            f'in {data_path}.'
        )
    n_digits = len(fl) - 9
    
    return n_digits
    
def make_pca(ad):
    sc.pp.pca(ad)
    return ad

def make_diffusion_components(ad, n_components=10):
    dm_res = palantir.utils.run_diffusion_maps(pd.DataFrame(ad.obsm['X_pca'], index=ad.obs_names), n_components=n_components)
    ad.obsp['DM_Kernel'] = dm_res['kernel']
    ad.obsm['DM_EigenVectors'] = dm_res['EigenVectors'].values
    ad.uns['DMEigenValues'] = dm_res['EigenValues'].values
    return ad

def profile_call(fun, out_path, name=None):
    if not name:
        name = fun.__name__
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = fun(*args, **kwargs)
        pr.disable()
        pr.dump_stats(out_path)
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        with open(out_path+'.log', 'w+') as f:
            f.write(s.getvalue())
        return result
    return wrapper

def warmup(ad):
    print('warmup pca...')
    make_pca(ad)
    print('warmup diffusion maps...')
    make_diffusion_components(ad)
    
def save_estimator_stats(ad, estimator):
    try:
        ls = estimator.ls.item()
    except AttributeError:
        ls = estimator.ls
    try:
        mu = estimator.mu.item()
    except AttributeError:
        mu = estimator.mu
    ad.uns['density_estimator_stats'] = {
        'n_landmarks':estimator.n_landmarks,
        'density_estimator_stats':None,
        'target_rank':estimator.rank,
        'final_rank':estimator.L.shape[1],
        'length_scale_fac':estimator.ls_factor,
        'length_scale':ls,
        'mu':mu,
        'n_obs':estimator.x.shape[0],
        'n_dims':estimator.x.shape[1],
        'method':estimator.method,
        'd_method':estimator.d_method,
        'd':estimator.d,
        'dm_comps':ad.obsm['DM_EigenVectors'].shape[1],
        'estimator':str(estimator),
    }
    if estimator.landmarks is not None:
        ad.uns['density_estimator_stats']["n_landmarks"] = estimator.landmarks.shape[0]
    if"leiden" in ad.uns.keys():
        ad.uns['density_estimator_stats']["leiden_resolution"] = ad.uns["leiden"]["params"]["resolution"]
    ad.uns['log_density_predictor'] = estimator.predict.to_dict()
        

def run_pipeline(run_density_estimation, ad, n_landmarks=5000):
    
    print('pca...')
    make_pca(ad)
    print('diffusion maps...')
    make_diffusion_components(ad)
    X = ad.obsm['DM_EigenVectors']
    run_density_estimation(ad, X, n_landmarks)
    
    
        
def main_stability(BASE_OUT_PATH, make_estimator, n_components=10):
    dataset, sub_id, n_landmarks = get_args()
    data_path = f'{BASE_IN_PATH}{dataset}/'
    
    
    print('loading data...')
    n_digits = get_digit_count(data_path)
    id_str = '0' * n_digits
    ad_path = data_path + f'sub_{id_str}.h5ad'
    ad = read_h5ad(ad_path)
    
    out_dir = f'{BASE_OUT_PATH}{dataset}'
    if not os.path.exists(out_dir):
        print(f'Making {out_dir}.')
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
    else:
        print(f'Using {out_dir}.')
        
    print('pca...')
    make_pca(ad)
    print('diffusion maps...')
    make_diffusion_components(ad, n_components=n_components)
    X = ad.obsm['DM_EigenVectors']
    print('density...')
    estimator = make_estimator(sub_id, ad, n_landmarks=5000)
    ad.obs['log_density'] = estimator.fit_predict(X)
    save_estimator_stats(ad, estimator)
    
    n_digits = 10
    n_id_str = f'{sub_id:0{n_digits}}'
    density_path = f'{out_dir}/log_densities_{n_id_str}.csv'
    ad.obs['log_density'].to_csv(density_path)
    est_stats_path = f'{out_dir}/estimator_stats_{n_id_str}.csv'
    pd.Series(ad.uns['density_estimator_stats']).to_csv(est_stats_path)
    predictor_path = f'{out_dir}/predictor_{n_id_str}.json'
    estimator.predict.to_json(predictor_path)
    print(f'Saved results to {out_dir}.')

def main(BASE_OUT_PATH, run_density_estimation):
    dataset, sub_id, n_landmarks = get_args()
    data_path = f'{BASE_IN_PATH}{dataset}/'
    n_digits = get_digit_count(data_path)
    id_str = f'{sub_id:0{n_digits}}'
    
    print('loading data...')
    ad_path = data_path + f'sub_{id_str}.h5ad'
    ad = read_h5ad(ad_path)

    warmup(ad)
    
    out_dir = f'{BASE_OUT_PATH}{dataset}'
    if not os.path.exists(out_dir):
        print(f'Making {out_dir}.')
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
    else:
        print(f'Using {out_dir}.')
    
    mem_dump_path = f'{out_dir}/profile_mem_dump_{id_str}.log'
    time_dump_path = f'{out_dir}/profile_time_dump_{id_str}'
    
    with open(mem_dump_path, "w+") as fp:
        profile_call(
            profile(run_pipeline, fp),
            time_dump_path,
        )(run_density_estimation, ad, n_landmarks)
    
    density_path = f'{out_dir}/log_densities_{id_str}.csv'
    ad.obs['log_density'].to_csv(density_path)
    est_stats_path = f'{out_dir}/estimator_stats_{id_str}.csv'
    pd.Series(ad.uns['density_estimator_stats']).to_csv(est_stats_path)
    print(f'Saved results to {out_dir}.')
