#!/bin/bash
set -eu

BASE_DIR=# working directory e.g. /export/data1/dliu/iwslt2023/mt
name=$1 # MT model name
sl=en
tl=$2 # target language
ckpt=$3 # checkpoint name
DICT=#Path to dictionary e.g. $BASE_DIR/model/deltalm/dict_multilingual.txt
FAIRSEQ_MODEL=$BASE_DIR/model/$name

path2data=# path to binarized test data
PRETRAINED_DELTALM=# path to pretrained deltalm model, needed when initializing models
EVAL_DIR=#where to write out results

lang_pairs="en-ar,en-zh,en-nl,en-fr,en-de,en-ja,en-fa,en-pt,en-ru,en-tr"

if [[ $name == *"adapter"* ]]; then
        ADAPTERS="--one-dataset-per-batch --enable-lang-ids"
else
        ADAPTERS=""
fi

CUDA_VISIBLE_DEVICES=0 python ./generate.py $path2data \
    --path $FAIRSEQ_MODEL/checkpoint_$ckpt.pt \
    --task translation_multi_simple_epoch \
    --lang-pairs $lang_pairs \
    --fixed-dictionary $DICT \
    --batch-size 512 --beam 5 --remove-bpe=sentencepiece \
    --source-lang $sl --target-lang $tl \
    --gen-subset test \
    --encoder-langtok "tgt" \
    --decoder-langtok $ADAPTERS \
    --skip-invalid-size-inputs-valid-test --fp16 \
    --model-overrides "{'pretrained_deltalm_checkpoint': $PRETRAINED_DELTALM}" > $EVAL_DIR/$sl-$tl.pred.log

grep ^H $EVAL_DIR/$sl-$tl.pred.log | cut -f3- | perl -nle 'print ucfirst' > $EVAL_DIR/$sl-$tl.sys
grep ^T $EVAL_DIR/$sl-$tl.pred.log | cut -f2- | perl -nle 'print ucfirst' > $EVAL_DIR/$sl-$tl.ref