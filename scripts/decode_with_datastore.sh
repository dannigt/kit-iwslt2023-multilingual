#!/bin/bash
set -eu
source ./knn_param.sh

BASE_DIR=# working directory e.g. /export/data1/dliu/iwslt2023/mt
name=$1 # MT model name
sl=en
tl=$2 # target language
ckpt=$3 # checkpoint name
FAIRSEQ_MODEL=$BASE_DIR/model/$name

path2data=# path to binarized test data

PRETRAINED_DELTALM=# path to pretrained deltalm model, needed when initializing models

w=0.3 # interpolation weight for kNN-MT
k=8 # number of neighbors
T=50 # temperature for kNN distribution

EVAL_DIR=#where to write out results e.g. $BASE_DIR/data/$name/$tst/$datastore_name/${ckpt}_knn.w${w}.k${k}.T${T}

# kNN parameters
DSTORE_SIZE=${tokmap[$tl]}
DATASTORE_PATH=$FAIRSEQ_MODEL/knn_datastore_$dstore_name # where the datastore was saved

# Only relevant when ensembling
ENSEMBLE="" # Path to other models for ensemgling ":$OTHER_MODEL_PATH"
ENSEMBLE_DATASTORE="" # Path to other models' datastores

if [[ $name == *"adapter"* ]]; then
        ADAPTERS="--one-dataset-per-batch --enable-lang-ids"
else
        ADAPTERS=""
fi

ln -snf ./deltalm_knn ./fairseq/  # hacky way to "import" deltalm_knn

CUDA_VISIBLE_DEVICES=0 python ./fairseq/experimental_generate.py $path2data \
    --path $FAIRSEQ_MODEL/checkpoint_$ckpt.pt$ENSEMBLE \
    --task translation_multi_simple_epoch \
    --lang-pairs $lang_pairs \
    --arch deltalm_large_with_datastore \
    --gen-subset test \
    --source-lang $sl --target-lang $tl \
    --encoder-langtok "tgt" \
    --decoder-langtok $ADAPTERS \
    --batch-size 64 --beam 5 --remove-bpe=sentencepiece --no-repeat-ngram-size 3 \
    --model-overrides "{'pretrained_deltalm_checkpoint': $PRETRAINED_DELTALM,
    'load_knn_datastore': True, 'use_knn_datastore': True,
    'dstore_filename': '$DATASTORE_PATH', 'extra_dstore_filename': '$ENSEMBLE_DATASTORE',
    'dstore_size': $DSTORE_SIZE, 'dstore_fp16': False, 'k': $k, 'probe': 32,
    'faiss_metric_type': 'do_not_recomp_l2', 'knn_k_type': 'fix', 'max_k': 32, 'label_count_as_feature': False, 'relative_label_count': False, 'avg_k': False,
    'knn_sim_func': 'do_not_recomp_l2', 'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True,
    'knn_lambda_type': 'fix', 'knn_lambda_value': $w, 'knn_temperature_type': 'fix', 'knn_temperature_value': $T,
     }"  > $EVAL_DIR/$sl-$tl.pred.log

grep ^H $EVAL_DIR/$sl-$tl.pred.log | cut -f3- | perl -nle 'print ucfirst' > $EVAL_DIR/$sl-$tl.sys
grep ^T $EVAL_DIR/$sl-$tl.pred.log | cut -f2- | perl -nle 'print ucfirst' > $EVAL_DIR/$sl-$tl.ref