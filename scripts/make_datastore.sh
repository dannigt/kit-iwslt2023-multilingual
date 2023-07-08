#!/bin/bash
set -eu
source ./knn_param.sh

BASE_DIR=# working directory e.g. /export/data1/dliu/iwslt2023/mt
name=$1 # MT model name
sl=en
tl=$2 # target language
ckpt=$3 # checkpoint name
FAIRSEQ_MODEL=$BASE_DIR/model/$name

BIN_DATA_PATH=# path to binarized data to use as datastore e.g. $PREPRO_DIR/bin/eval/$dstore_name
ncentroids=128

ln -snf ./deltalm ./fairseq/  # hacky way to "import" deltalm

# kNN parameters
DSTORE_SIZE=${tokmap[$tl]} # number of target tokens
DATASTORE_PATH=$FAIRSEQ_MODEL/knn_datastore_$dstore_name # where to save datastore

lang_pairs="en-ar,en-zh,en-nl,en-fr,en-de,en-ja,en-fa,en-pt,en-ru,en-tr"
DICT=#Path to dictionary e.g. $BASE_DIR/model/deltalm/dict_multilingual.txt

DATASTORE_PATH=$FAIRSEQ_MODEL/$tl/knn_datastore_$dstore_name
mkdir -p $DATASTORE_PATH

if [[ $name == *"adapter"* ]]; then # additional parameters for adapter
        ADAPTERS="--one-dataset-per-batch --enable-lang-ids"
else
        ADAPTERS=""
fi

CUDA_VISIBLE_DEVICES=0 python ./fairseq/save_datastore.py $BIN_DATA_PATH \
    --dataset-impl mmap \
    --task translation_multi_simple_epoch \
    --lang-pairs $lang_pairs \
    --fixed-dictionary $DICT \
    --valid-subset test \
    --path $FAIRSEQ_MODEL/checkpoint_$ckpt.pt \
    --encoder-langtok "tgt" \
    --decoder-langtok \
    -s $sl -t $tl \
    --batch-size 1024 --skip-invalid-size-inputs-valid-test $ADAPTERS \
    --decoder-embed-dim 1024 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH

CUDA_VISIBLE_DEVICES=0 python ./fairseq/train_datastore_gpu.py \
  --dstore_mmap $DATASTORE_PATH \
  --dstore_size $DSTORE_SIZE \
  --faiss_index ${DATASTORE_PATH}/knn_index \
  --ncentroids $ncentroids \
  --probe 32 \
  --dimension 1024

echo "finished making datastore for language:" $tl