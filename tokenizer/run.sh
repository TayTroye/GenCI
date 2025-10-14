export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1


DATA_LIST=()


for DATA in ${DATA_LIST[@]}; do

    python main.py \
        --dataset=$DATA \
        --config_file=config.yaml \
        --ckpt_name=rqvae


    python generate_tokens.py \
        --dataset=$DATA \
        --config_file=config.yaml \
        --ckpt_name=rqvae\
        --epoch_ckpts 9994 9995 9996 9997 9998 9999 10000

done

