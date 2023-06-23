#! /bin/bash -

in_path='/fh/fast/setty_m/user/dotto/CellDensities/data/subsamples/hierarchical'
eval "$(micromamba shell hook --shell=bash)"
micromamba activate mellon_v2

ls "$in_path" | while read ds_name; do
    ds_path="$in_path/$ds_name"

    sbatch --array=0-200 ./run_stability_ls.py $ds_name
    sbatch --array=0-200 ./run_stability_ls_noNy.py $ds_name
    sbatch --array=0-200 ./run_stability_n_landmarks.py $ds_name
    sbatch --array=0-200 ./run_stability_n_landmarks_noNy.py $ds_name
    sbatch --array=0-200 ./run_stability_rank.py $ds_name
    sbatch --array=1-100 ./run_stability_dmc_noNy.py $ds_name
    sbatch --array=1-100 ./run_stability_dmc_noNy_fractal.py $ds_name
    sbatch --array=0-200 ./run_stability_res_leiden_noNy.py $ds_name
    sbatch --array=0-200 ./run_stability_n_maxmin_noNy.py $ds_name
    sbatch --array=0-100 ./run_stability_d_noNy.py $ds_name
done
