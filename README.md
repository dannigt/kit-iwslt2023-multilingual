# KIT's Multilingual Speech Translation System for IWSLT 2023

System description [paper](https://arxiv.org/pdf/2306.05320.pdf) to appear in IWSLT 2023.

## MT System

The implementation is based on the following repos:

* [DeltaLM](https://github.com/microsoft/unilm/tree/master/deltalm) by [Ma et al., (2021)](https://arxiv.org/pdf/2106.13736.pdf)

* [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt) by [Zheng et al., (2021)](https://arxiv.org/pdf/2105.13022.pdf)

* [hyperadapters](https://github.com/cbaziotis/fairseq/tree/hyperadapters/examples/adapters) by [Baziotis et al., (2022)](https://arxiv.org/pdf/2205.10835.pdf)

### Requirements
To use kNN-MT, set up according to the dependency versions [here](https://github.com/zhengxxn/adaptive-knn-mt#requirements-and-installation).   

### Data
The multilingual system was trained on ~570M sentence pairs (after data augmentation).

As data of this volume is difficult to host, we provide the [binarized data](https://bwsyncandshare.kit.edu/s/Gow9Z6X7QNMtrwy) (can be directly used for fairseq).
For the original data, please open an issue or reach out by mail, so we could set up individual data transfers.  

The [dictionary](https://bwsyncandshare.kit.edu/s/eDbnDSLBdkMga6D) and [SPM model](https://github.com/microsoft/unilm/tree/master/deltalm#pretrained-models) are the same as for DeltaLM.
We added language tags at the end of the dictionary.

### Models

#### Baseline Multilingual Model

For training:
```
bash ./scripts/train_baseline.sh
```
For decoding:
```
bash ./scripts/decode.sh $MT_MODEL_NAME $TARGET_LANG $CKPT_NAME  
```

| Model                                | Download | Results            |
|--------------------------------------|----------|--------------------|
| deltaLM trained on diversified data  | [ckpt](https://bwsyncandshare.kit.edu/s/7Jb3Zot3mGJLemk) | Row (2) of Table 7 |


Note the models are trained on true-cased data. For inference, we need to upper case the first letter for languages with casing.

#### kNN-MT


##### Create datastore

To create a data store from a trained MT model:
```
bash ./scripts/make_datastore.sh $MT_MODEL_NAME $TARGET_LANG $CKPT_NAME  
```

There are also helpful documentations from [adaptive-knn-mt](https://github.com/zhengxxn/adaptive-knn-mt#create-datastore).

##### Run inference with kNN datastore
```
bash ./scripts/decode_with_datastore.sh $MT_MODEL_NAME $TARGET_LANG $CKPT_NAME  
```

#### Adapters
```
bash ./scripts/train_adapter.sh
```

## ASR System

We do not have the original WavLM + mBART ASR system anymore.

A system with a similar architecture (WavLM + BART) trained using additional data is [here](https://huggingface.co/nguyenvulebinh/wavlm-bart). 

## End-to-End Speech Translation System

We use [NMTGMinor](https://github.com/quanpn90/NMTGMinor) (included as submodule here) to train the e2e systems.

We share the trained models: 

| Model                        | Download | Results            |
|------------------------------|----------|--------------------|
| wavLM+mBART50                | [ckpt](https://bwsyncandshare.kit.edu/s/HGGBoGfaHbdnRfG) | Row (6) of Table 7 |
| wavLM+mBART50 with TTS data  | [ckpt](https://bwsyncandshare.kit.edu/s/3mfJQ97jmXyHW2n) | Row (7) of Table 7 |

For training, we first need to download pretrained  models into where we call the training script:
* WavLM: we use the large model ([download](https://github.com/microsoft/unilm/tree/master/wavlm#pre-trained-models))
* mBART tokenizers etc. ([download](https://bwsyncandshare.kit.edu/s/Af94THjjxKgxyMn))

To binarize the data, for each partition (dataset and language pair), we need `.seg` and `.label` files, which are the source speech and target texts. For example:

```bash
 |-dev
 | |-MuST-C.en-de.label
 | |-MuST-C.en-de.seg
 | |-...
 |-train
 | |-MuST-C.en-de.label
 | |-MuST-C.en-de.seg
 | |-...
```

For the `.seg` files, the 3rd and 4rd columns are start and end timestamps of the utterance.

```bash
head -3 MuST-C.en-de.seg
ted_1_0 PATH_TO_DATA/MuST-C/en-de/data/train/wav/ted_1.wav 28.05 29.13
ted_1_1 PATH_TO_DATA/MuST-C/en-de/data/train/wav/ted_1.wav 29.92 36.01
ted_1_2 PATH_TO_DATA/MuST-C/en-de/data/train/wav/ted_1.wav 36.40 47.74

head -3 MuST-C.en-de.label
Vielen Dank, Chris.
Es ist mir wirklich eine Ehre, zweimal auf dieser Bühne stehen zu dürfen. Tausend Dank dafür.
Ich bin wirklich begeistert von dieser Konferenz, und ich danke Ihnen allen für die vielen netten Kommentare zu meiner Rede vorgestern Abend.
```

For binarization:
```bash
bash ./scripts/prepro_e2e.sh
```

For training:
```bash
bash ./scripts/train_e2e.sh
```

For inference:
```bash
bash ./scripts/decode_e2e.sh
```