#!/bin/bash

GPU=0,1,2,3
path_2_data=#path to binarized data
deltalm="large" # DeltaLM configuration
PRETRAINED_MODEL=#path to pretrained deltalm model e.g. $HOME/projects/iwslt23/model/deltalm/deltalm-$deltalm.pt
save_dir=#path to save model ckpt
lr=0.0001
lang_pairs="en-ar,en-zh,en-nl,en-fr,en-de,en-ja,en-fa,en-pt,en-ru,en-tr"

batch_size=2048 # should fit for A100 40GB

CUDA_VISIBLE_DEVICES=$GPU python ./train.py $path_2_data \
    --save-dir $save_dir \
    --arch deltalm_$deltalm \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --task translation_multi_simple_epoch \
    --lang-pairs $lang_pairs \
    --share-all-embeddings \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $lr \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 4000 \
    --max-update 400000 \
    --max-epoch 100 \
    --max-tokens $batch_size \
    --update-freq 40 \
    --seed 1 \
    --no-epoch-checkpoints \
    --dropout 0.1 \
    --attention-dropout 0 \
    --encoder-langtok "tgt" \
    --decoder-langtok \
    --sampling-method "temperature" \
    --sampling-temperature 5.0 \
    --validate-interval-updates 1000 \
    --log-interval 50 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 1 \
    --skip-invalid-size-inputs-valid-test \
    --save-interval-updates 1000 \
    --fp16 2>&1 | tee -a $save_dir/train.log