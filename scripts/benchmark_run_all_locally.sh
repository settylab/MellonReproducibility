#! /bin/bash -

in_path='/fh/fast/setty_m/user/dotto/CellDensities/data/subsamples/hierarchical'
eval "$(micromamba shell hook --shell=bash)"
micromamba activate cestiny-16

ls "$in_path" | while read ds_name; do
    echo "###### Dataset $ds_name"
    ds_path="$in_path/$ds_name"
    n=$(ls "$ds_path" | wc -l)
    n_landmarks=5000
    #[[ "$ds_name" == ips ]] && n_landmarks=20000
    
    for i in $(seq 0 $(( n - 1 ))); do
        export SLURM_ARRAY_TASK_ID=$i
        echo "# Dataset $ds_name run number $(( i + 1)) of $n"
        python run_subsample_noNy_fractal.py $ds_name $n_landmarks
        python run_subsample_noNy.py $ds_name $n_landmarks
        python run_subsample.py $ds_name $n_landmarks
        python run_subsample_noNy_1core.py $ds_name $n_landmarks
        python run_subsample_1core.py $ds_name $n_landmarks
    done
done
