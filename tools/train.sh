#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/levir_cd.yaml
labeled_id_path=partitions/levircd/1_10/train_l.txt
unlabeled_id_path=partitions/levircd/1_10/train_u.txt
save_path=/data/home/jinjuncan/dataset/exp_levir/exp2

mkdir -p $save_path

export CUDA_VISIBLE_DEVICES=6,7

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    unimatch.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt