#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import os
import sys
from argparse import Namespace
from itertools import chain

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from tqdm import tqdm
from fairseq.utils import reset_logging
from omegaconf import DictConfig

import deltalm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


# ------ add by
# this script is implemented based on validate.py, and refers to the implementation of knnlm
# we only need to go through the dataset like in training, and save the datastore
# ------

def main(args: DictConfig, override_args=None):

    if isinstance(args, Namespace):
        args = convert_namespace_to_omegaconf(args)

    utils.import_user_module(args.common)

    reset_logging()

    assert (
            args.dataset.max_tokens is not None or args.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"


    logger.info(args)

    use_fp16 = args.common.fp16
    use_cuda = torch.cuda.is_available() and not args.common.cpu
   
    if use_cuda:
        torch.cuda.set_device(args.distributed_training.device_id)

    if args.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None


#    task = tasks.setup_task(args.task)
#
#    # Set dictionaries
#    try:
#        src_dict = getattr(task, "source_dictionary", None)
#    except NotImplementedError:
#        src_dict = None
#    tgt_dict = task.target_dictionary
#
#    overrides = ast.literal_eval(args.common_eval.model_overrides)
#
    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.common_eval.path))
   # models, saved_cfg = checkpoint_utils.load_model_ensemble(
   #     utils.split_paths(args.common_eval.path),
   #     arg_overrides=overrides,
   #     task=task,
   #     suffix=args.checkpoint.checkpoint_suffix,
   #     strict=(args.checkpoint.checkpoint_shard_count == 1),
   #     num_shards=args.checkpoint.checkpoint_shard_count,
   # )
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.common_eval.path],
        arg_overrides=overrides,
        suffix=args.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

#    logger.info("loading model(s) from {}".format(args.common_eval.path))
#    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
#        [args.common_eval.path],
#        arg_overrides=overrides,
#        suffix=args.checkpoint.checkpoint_suffix,
#    )
#    model = models[0]
#
#    # Move models to GPU
#    for model in models:
#        model.eval()
#        if use_fp16:
#            model.half()
#        if use_cuda:
#            model.cuda()
#
#    # Print args
#    logger.info(model_args)
#
    # Build criterion, we do not need this, so remove it, by
    # criterion = task.build_criterion(model_args)
    # criterion.eval()
#    if args.save_plain_text:
#        batch_src_tokens = []
#        batch_target = []
#
    # --- check save data store , add by
    import numpy as np
    if args.model.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.model.dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+',
                                shape=(args.model.dstore_size, args.model.decoder_embed_dim))
        dstore_vals = np.memmap(args.model.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.model.dstore_size, 1))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.model.dstore_mmap + '/keys.npy', dtype=np.float32, mode='w+',
                                shape=(args.model.dstore_size, args.model.decoder_embed_dim))
        dstore_vals = np.memmap(args.model.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.model.dstore_size, 1))

    dstore_idx = 0
    # --- end
    
    data_idx = 1

    for subset in args.dataset.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset(subset)
            #task.args.required_seq_len_multiple = 1
            #task.args.load_alignments = False
            #task.load_dataset(subset, combine=False, epoch=data_idx)
            #data_idx = data_idx + 1
            #dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.dataset.max_tokens,
            max_sentences=args.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.dataset.required_batch_size_multiple,
            seed=args.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.dataset.num_workers,
            data_buffer_size=args.dataset.data_buffer_size,
        ).next_epoch_itr(False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.common.log_format,
            log_interval=args.common.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.common.no_progress_bar else "simple"),
        )

        log_outputs = []
   
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                
                # for adapters
                src_tokens = sample["net_input"]["src_tokens"]
                src_lang_id = task.data_manager.lang_dict.indices[task.source_langs[0]]
                tgt_lang_id = task.data_manager.lang_dict.indices[task.target_langs[0]]
                print("==========lang IDs:", src_lang_id, tgt_lang_id)

                sample["net_input"]["src_lang_id"] = src_tokens.new([src_lang_id]).repeat(src_tokens.size(0), 1)
                sample["net_input"]["tgt_lang_id"] = src_tokens.new([tgt_lang_id]).repeat(src_tokens.size(0), 1)

                sample = utils.move_to_cuda(sample) if use_cuda else sample

                # -------- add by , we should go through the model with the sample and get the hidden state
                # so we append a forward_and_get_hidden_state_step method in Translation task
                features = task.forward_and_get_hidden_state_step(sample, model)  # [B, T, H]
                target = sample['target']  # [B, T]
                # print('feature_size:{}'.format(features.size()))
                # print('target_size:{}'.format(target.size()))

                # get useful parameters
                batch_size = target.size(0)
                seq_len = target.size(1)
                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]

                # remove the pad tokens and related hidden states
                target = target.view(batch_size * seq_len)
                target_mask = target_mask.view(batch_size * seq_len)

                non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                target = target.index_select(dim=0, index=non_pad_index)  # [n_count]

                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                # if save plain text
#                if args.save_plain_text:
#                    src_tokens = sample['net_input']['src_tokens']  # [B, src len]
#                    assert src_tokens.size(0) == batch_size
#                    src_len = src_tokens.size(-1)
#                    src_tokens = src_tokens.unsqueeze(1).expand(batch_size, seq_len, src_len). \
#                        reshape((batch_size * seq_len, src_len))
#                    src_tokens = src_tokens.index_select(dim=0, index=non_pad_index)  # [n_count, src_len]
#                    batch_src_tokens.append(src_tokens.cpu())
#                    batch_target.append(target.cpu().unsqueeze(-1))
#
                # save to the dstore
                current_batch_count = target.size(0)
                if dstore_idx + current_batch_count > args.model.dstore_size:
                    reduce_size = args.model.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
#                    if args.save_plain_text:
#                        src_tokens = src_tokens[:reduce_size, :]
                else:
                    reduce_size = current_batch_count

                if args.model.dstore_fp16:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float16)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)
                else:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float32)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)

                # if args.save_plain_text:
                #     batch_src_tokens.append(src_tokens.cpu())
                #     batch_target.append(target.cpu())

                    # we need look up the dict
                    # TODO, here src strs is not debpe
                    # src_strs = src_dict.string(src_tokens, return_list=True)  # [[str]]
                    # trg_tokens = tgt_dict.string(target, return_list=True)  # [[token]]
                    # cur_trg_str = ""
                    # for src_str, trg_token in zip(src_strs, trg_tokens):
                    #     if len(trg_token) == 0:
                    #         _trg_token = "<eos>"
                    #     else:
                    #         _trg_token = trg_token
                    #     cur_trg_str = cur_trg_str + ' {}'.format(_trg_token)
                    #     plain_text.append("src: {} trg: {}".format(src_str, cur_trg_str))
                    #     if len(trg_token) == 0:
                    #         cur_trg_str = ""

                dstore_idx += reduce_size

                print(dstore_idx)
                if dstore_idx >= args.model.dstore_size:
                    print('much more than dstore size break')
                    break
            # -------- end, by

            # _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            # progress.log(log_output, step=i)
            # log_outputs.append(log_output)

        # if args.distributed_world_size > 1:
        #     log_outputs = distributed_utils.all_gather_list(
        #         log_outputs,
        #         max_size=getattr(args, "all_gather_list_size", 16384),
        #     )
        #     log_outputs = list(chain.from_iterable(log_outputs))

        # with metrics.aggregate() as agg:
        #     task.reduce_metrics(log_outputs, criterion)
        #     log_output = agg.get_smoothed_values()
        #
        # progress.print(log_output, tag=subset, step=i)

#    if args.save_plain_text:
#
#        # Set dictionaries
#        try:
#            src_dict = getattr(task, "source_dictionary", None)
#        except NotImplementedError:
#            src_dict = None
#        tgt_dict = task.target_dictionary
#
#        plain_text = []
#
#        for src_tokens, target in tqdm(zip(batch_src_tokens, batch_target)):
#
#            src_strs = src_dict.string(src_tokens, return_list=True, extra_symbols_to_ignore=[src_dict.pad()])  # [[str]]
#            trg_tokens = tgt_dict.string(target, return_list=True)  # [[token]]
#            cur_trg_str = ""
#
#            for src_str, trg_token in zip(src_strs, trg_tokens):
#                if len(trg_token) == 0:
#                    _trg_token = "<eos>"
#                else:
#                    _trg_token = trg_token
#                cur_trg_str = cur_trg_str + ' {}'.format(_trg_token)
#                plain_text.append("src: {} trg: {}".format(src_str, cur_trg_str))
#                if len(trg_token) == 0:
#                    cur_trg_str = ""
#
#        with open(args.model.dstore_mmap + '/text.txt', 'w') as f:
#            for line in plain_text:
#                f.write(f"{line}\n")
#

def cli_main():
    parser = options.get_save_datastore_parser()

    parser.add_argument(
        '--arch', '-a', metavar='ARCH', default="transformer",
        help='Model architecture. For constructing tasks that rely on '
             'model args (e.g. `AudioPretraining`)'
    )

    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)
#    distributed_utils.call_main(args, main, override_args=override_args)
    distributed_utils.call_main(
        convert_namespace_to_omegaconf(args), main, override_args=override_args
    )


if __name__ == "__main__":
    cli_main()
