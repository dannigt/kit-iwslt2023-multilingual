#!/bin/bash
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

set -eu

onmt=./NMTGMinor
orig_data=#original data directory

train_src=""
train_tgt=""
train_langs=""
train_langs_tgt=""

valid_src=""
valid_tgt=""
valid_langs=""
valid_langs_tgt=""

SAVEDIR=# where to save binarized data
mkdir -p $SAVEDIR

declare -A tokmap=( ["de"]="de_DE" ["ar"]="ar_AR" ["zh"]="zh_CN" ["nl"]="nl_XX" ["fr"]="fr_XX" ["ja"]="ja_XX" ["fa"]="fa_IR" ["pt"]="pt_XX" ["ru"]="ru_RU" ["tr"]="tr_TR")

for lang in de ar zh nl fr ja fa pt ru tr; do
for dataset in "MuST-C"; do
    pair="en-$lang"
    train_src=$train_src"$orig_data/train/$dataset.$pair.seg|"
    train_tgt=$train_tgt"$orig_data/train/$dataset.$pair.label|"
    train_langs=$train_langs"en_XX|"
    train_langs_tgt=$train_langs_tgt"${tokmap[$lang]}|"

    valid_src=$valid_src"$orig_data/dev/$dataset.$pair.seg|"
    valid_tgt=$valid_tgt"$orig_data/dev/$dataset.$pair.label|"
    valid_langs=$valid_langs"en_XX|"
    valid_langs_tgt=$valid_langs_tgt"${tokmap[$lang]}|"
done
done

train_src=${train_src::-1}
train_tgt=${train_tgt::-1}
train_langs=${train_langs::-1}
train_langs_tgt=${train_langs_tgt::-1}

valid_src=${valid_src::-1}
valid_tgt=${valid_tgt::-1}
valid_langs=${valid_langs::-1}
valid_langs_tgt=${valid_langs_tgt::-1}

echo $train_langs $train_langs_tgt $valid_langs $valid_langs_tgt
threads=1

#OMP_NUM_THREADS=1
python3 -u $onmt/preprocess.py -train_src $train_src -format wav -multi_dataset \
        -train_tgt $train_tgt \
        -valid_src $valid_src \
        -valid_tgt $valid_tgt \
        -train_src_lang $train_langs -train_tgt_lang $train_langs_tgt \
        -valid_src_lang $valid_langs -valid_tgt_lang $valid_langs_tgt \
        -src_seq_length 9999999 \
        -tgt_seq_length 9999999 \
        -concat 1 -asr -src_type audio -asr_format wav -tgt_pad_token '<pad>'  \
        -save_data $SAVEDIR/data -num_threads $threads \
        -external_tokenizer "facebook/mbart-large-50" -tgt_vocab ./vocab.txt