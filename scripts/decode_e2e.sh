#!/bin/bash
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONWARNINGS="ignore"

declare -A tokmap=( ["de"]="de_DE" ["ar"]="ar_AR" ["zh"]="zh_CN" ["nl"]="nl_XX" ["fr"]="fr_XX" ["ja"]="ja_XX" ["fa"]="fa_IR" ["pt"]="pt_XX" ["ru"]="ru_RU" ["tr"]="tr_TR")

for tl in zh ja ar nl fr fa pt ru tr; do

        lang=${tokmap[$tl]}
        input=$EVALDIR/en-$tl.en  # source .seg file
        hyp=$OUTDIR/$sl-$tl.sys # output file
        python -u $onmt/translate_distributed.py -model $mymodel  -verbose -fast_translate  -fp16 -no_buffering \
                                        -gpus $GPU \
                                        -src $input \
                                        -asr_format "wav" -external_tokenizer "facebook/mbart-large-50" \
                                        -output $hyp \
                                        -beam_size $beam \
                                        -concat 1 -no_repeat_ngram_size 0 \
                                        -normalize -alpha $alpha \
                                        -encoder_type audio -verbose \
                                        -src_lang "en_XX" -tgt_lang $lang -bos_token $lang \
                                        -batch_size $bsz

        cat $hyp | perl -nle 'print ucfirst' > $hyp.ucfirst

done