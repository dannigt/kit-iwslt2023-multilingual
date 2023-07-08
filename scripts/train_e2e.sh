#!/bin/bash
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

onmt=./NMTGMinor
dataset="all_st_fwdtrans" # preprocessed dataset name

BASEDIR=#experiment working dir
path2data=#path to data

GPU=0,1,2,3
name=wavlm.mbart
MODELDIR=$BASEDIR/model/st/$name

# Batch size frames and words
bszf=1280000
bszw=2048

mkdir -p $MODELDIR

DEC_CONFIG_FILE=#path to mbart config, get from https://huggingface.co/facebook/mbart-large-50/blob/main/config.json

CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 python3 -u  $onmt/train_distributed.py \
                                -data $path2data -buffer_size 8 \
                                -data_format wav  -gpus 0 1 2 3  -fp16 -reset_optim \
                                -save_model $MODELDIR/model -find_unused_parameter -freeze_embedding \
                                -model wav2vec2_bert \
                                -batch_size_words $bszw -batch_size_frames $bszf \
                                -update_frequency 32 \
                                -batch_size_sents 9999999999 -max_src_length 512000 \
                                -batch_size_multiplier 8 \
                                -patch_vocab_multiplier 1 \
                                -max_src_length 512000 -max_tgt_length 512 \
                                -encoder_type wav2vec2 \
                                -enc_pretrained_model "wavlm" \
                                -dec_pretrained_model "mbart50" \
                                -src_pad_word "<pad>" \
                                -tgt_pad_word "<pad>" \
                                -dec_config_file $DEC_CONFIG_FILE \
                                -wav2vec2_pretrained_model 'WavLM-Large.pt' \
                                -input_size 1 \
                                -layers 6 -tie_weights \
                                -encoder_layers 24 -death_rate 0.1 -death_rate_decoder 0.1 -run_validation_before_training \
                                -model_size 1024 \
                                -inner_size 4096 \
                                -n_heads 16 \
                                -dropout 0.0 -residual_dropout 0.0 -ffn_dropout 0.3 -enc_pretrain_hidden_dropout 0.0 \
                                -attn_dropout 0.0 \
                                -emb_dropout 0.0 \
                                -word_dropout 0.0 \
                                -label_smoothing 0.1 \
                                -epochs 1000  -max_grad_norm 1 \
                                -learning_rate 1 -warmup_steps 4096 -weight_decay 0.001 -save_metrics accuracy  \
                                -optim 'adam' \
                                -update_method 'noam' \
                                -seed 171 \
                                -log_interval 1000 -save_every 1000 -keep_save_files 10 -multi_dataset \
                                -fp16 2>&1 | tee $MODELDIR/train.log