MODEL="AutoCode"
CONFIG="yaml/genci_ml.yaml"
DATASET="movielens"


python run_recbole.py \
    --model=$MODEL \
    --config_files=$CONFIG \
    --dataset=$DATASET



CONFIG="yaml/genci_instrument.yaml"
DATASET="instrument"

python run_recbole.py \
    --model=$MODEL \
    --config_files=$CONFIG \
    --dataset=$DATASET

