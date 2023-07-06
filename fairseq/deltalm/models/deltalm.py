# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayerBase
)
from fairseq.modules import LayerNorm
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq import utils

import math
from fairseq.modules.knn_datastore import KNN_Dstore
import torch.nn.functional as functional

import logging
logger = logging.getLogger(__name__)


def upgrade_state_dict_for_deltalm(
    state_dict: Dict[str, Any], pretrained_deltalm_checkpoint: str, is_encoder=True,
) -> Dict[str, Any]:

    if not os.path.exists(pretrained_deltalm_checkpoint):
       raise IOError("Model file not found: {}".format(pretrained_deltalm_checkpoint))
    
    with open(pretrained_deltalm_checkpoint, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    deltalm_state_dict = state["weights"]

    new_deltalm_state_dict = {}

    for key in deltalm_state_dict.keys():
        if is_encoder:
            if key.startswith('encoder.') or key.startswith('src_embedding.'):
                new_key = key.replace('encoder.', '')
                new_key = new_key.replace('src_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]
        else:
            if key.startswith('decoder.') or key.startswith('tgt_embedding.'):
                new_key = key.replace('decoder.', '')
                new_key = new_key.replace('tgt_embedding.', '')
                new_deltalm_state_dict[new_key] = deltalm_state_dict[key]

    deltalm_state_dict = new_deltalm_state_dict

    for key in deltalm_state_dict.keys():
        map_key = key
        map_key = map_key.replace('.ffn_1.fc1', '.fc3')
        map_key = map_key.replace('.ffn_1.fc2', '.fc4')
        map_key = map_key.replace('.ffn_2', '')
        map_key = map_key.replace('.ffn.', '.')
        map_key = map_key.replace('emb_layer_norm', 'layernorm_embedding')
        assert map_key in state_dict, map_key
        if 'embed_positions' in key or 'embed_tokens' in key:
            left_size = state_dict[map_key].size(0)
            right_size = deltalm_state_dict[key].size(0)
            if left_size <= right_size:
                state_dict[map_key] = deltalm_state_dict[key][:left_size]
            else:
                state_dict[map_key][:right_size] = deltalm_state_dict[key]
        else:
            state_dict[map_key] = deltalm_state_dict[key]

    return state_dict


@register_model("deltalm")
class DeltaLMModel(TransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-deltalm-checkpoint",
            type=str,
            metavar="STR",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMEncoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMDecoderDatastore(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)
        # return DeltaLMDecoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)

@register_model("deltalm_with_datastore")
class DeltaLMDatastore(DeltaLMModel):
    def load_state_dict(self, state_dict, strict=True, args=None):
        """we rewrite the load state dict here for only load part of trained model
        add by
        """
        if self.decoder.knn_lambda_type == 'trainable' or self.decoder.knn_temperature_type == 'trainable' \
                or self.decoder.use_knn_datastore:

            self.upgrade_state_dict(state_dict)
            from fairseq.checkpoint_utils import prune_state_dict
            new_state_dict = prune_state_dict(state_dict, args)

            print('-----------------knn load part of model-----------------')
            model_dict = self.state_dict()

            remove_keys = []
            for k, v in new_state_dict.items():
                if k not in model_dict:
                    remove_keys.append(k)

            for k in remove_keys:
                new_state_dict.pop(k)

            model_dict.update(new_state_dict)
            return super().load_state_dict(model_dict)

        else:
            return super().load_state_dict(state_dict, strict, args)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DeltaLMDecoderDatastore(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)


class DeltaLMEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=True,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            logger.info("Load DeltaLM's encoder from {0}".format(args.pretrained_deltalm_checkpoint))

class DeltaLMDecoder(TransformerDecoderBase):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, "pretrained_deltalm_checkpoint", "") != "":
            deltalm_loaded_state_dict = upgrade_state_dict_for_deltalm(
                state_dict=self.state_dict(),
                pretrained_deltalm_checkpoint=args.pretrained_deltalm_checkpoint,
                is_encoder=False,
            )
            self.load_state_dict(deltalm_loaded_state_dict, strict=True)
            logger.info("Load DeltaLM's decoder from {0}".format(args.pretrained_deltalm_checkpoint))

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DeltaLMDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer


class DeltaLMDecoderDatastore(DeltaLMDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.fp16 = args.fp16
        self.knn_datastore = None

        # for knn and ensemble
        self.updated_knn_datastore = False
        self.args = args
        self.dict_len = len(dictionary)

        if args.load_knn_datastore:
            # TODO: update to load corresponding one in ensemble order
            self.knn_datastore = KNN_Dstore(args, len(dictionary))

        self.use_knn_datastore = args.use_knn_datastore
        self.knn_lambda_type = args.knn_lambda_type
        self.knn_temperature_type = args.knn_temperature_type
        self.knn_k_type = args.knn_k_type
        self.label_count_as_feature = args.label_count_as_feature
        self.relative_label_count = args.relative_label_count
        self.avg_k = args.avg_k

        if self.knn_lambda_type == "trainable" and self.knn_k_type == "trainable":

            # TODO another network to predict k and lambda at the same time without gumbel softmax
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(args.max_k if not self.label_count_as_feature else args.max_k * 2,
                          args.k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=args.k_lambda_net_dropout_rate),
                nn.Linear(args.k_lambda_net_hid_size, 2 + int(math.log(args.max_k, 2))),
                nn.Softmax(dim=-1),  # [0 neighbor prob, 1 neighbor prob, 2 neighbor prob, 4 , 8 , ... , ]
            )

            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : args.k], gain=0.01)

            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, args.k:], gain=0.1)

        else:
            if self.knn_lambda_type == 'trainable':
                # TODO, we may update the label count feature here
                self.knn_distances_to_lambda = nn.Sequential(
                    nn.Linear(args.k if not self.label_count_as_feature else args.k * 2, args.knn_lambda_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_net_dropout_rate),
                    nn.Linear(args.knn_lambda_net_hid_size, 1),
                    nn.Sigmoid())

                if self.label_count_as_feature:
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, :args.k], mean=0, std=0.01)
                    # nn.init.normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], mean=0, std=0.1)

                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, : args.k], gain=0.01)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[0].weight[:, args.k:], gain=0.1)
                    nn.init.xavier_normal_(self.knn_distances_to_lambda[-2].weight)

                else:
                    nn.init.normal_(self.knn_distances_to_lambda[0].weight, mean=0, std=0.01)

            if self.knn_temperature_type == 'trainable':
                # TODO, consider a reasonable function
                self.knn_distance_to_temperature = nn.Sequential(
                    nn.Linear(args.k + 2, args.knn_temperature_net_hid_size),
                    nn.Tanh(),
                    nn.Linear(args.knn_temperature_net_hid_size, 1),
                    nn.Sigmoid())
                # the weight shape is [net hid size, k + 1)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, :-1], mean=0, std=0.01)
                nn.init.normal_(self.knn_distance_to_temperature[0].weight[:, -1:], mean=0, std=0.1)

            # TODO we split the network here for different function, but may combine them in the future
            if self.knn_k_type == "trainable":

                self.knn_distance_to_k = nn.Sequential(
                    nn.Linear(args.max_k * 2 if self.label_count_as_feature else args.max_k,
                              args.knn_k_net_hid_size),
                    nn.Tanh(),
                    nn.Dropout(p=args.knn_k_net_dropout_rate),
                    # nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Linear(args.knn_k_net_hid_size, args.max_k),
                    nn.Softmax(dim=-1))

                # nn.init.xavier_uniform_(self.knn_distance_to_k[0].weight, gain=0.01)
                # nn.init.xavier_uniform_(self.knn_distance_to_k[-2].weight, gain=0.01)
                # # TODO this maybe change or remove from here
                if self.label_count_as_feature:
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, :args.max_k], mean=0, std=0.01)
                    nn.init.normal_(self.knn_distance_to_k[0].weight[:, args.max_k:], mean=0, std=0.1)
                else:
                    nn.init.normal_(self.knn_distance_to_k[0].weight, mean=0, std=0.01)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        src_lang_id = None,
        tgt_lang_id = None
    ):

        # update datastore path for additional models in ensemble 
        if not self.updated_knn_datastore and self.ensemble_idx > 0:
            print(self.ensemble_idx)
            print("Old path:", self.args.dstore_filename)
            print("Updating datastore path for ensembling...")
            self.args.dstore_filename = self.args.extra_dstore_filename
            print("New path:", self.args.dstore_filename)
            self.knn_datastore = KNN_Dstore(self.args, self.dict_len)
            self.updated_knn_datastore = True

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )

        if self.use_knn_datastore:
            last_hidden = x

        if not features_only:
            x = self.output_layer(x)

        if self.use_knn_datastore:
            # we should return the prob of knn search
            knn_search_result = self.knn_datastore.retrieve(last_hidden)
            # knn_probs = knn_search_result['prob']
            knn_dists = knn_search_result['distance']  # [batch, seq len, k]  # we need do sort
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']

            if self.label_count_as_feature:
                # TODO, we get the segment label count here, which is conflict with previous experiment
                label_counts = self.knn_datastore.get_label_count_segment(tgt_index, relative=self.relative_label_count)
                network_inputs = torch.cat((knn_dists.detach(), label_counts.detach().float()), dim=-1)
            else:
                network_inputs = knn_dists.detach()

            if self.fp16:
                network_inputs = network_inputs.half()

            if self.knn_temperature_type == 'trainable':
                knn_temperature = None
            else:
                knn_temperature = self.knn_datastore.get_temperature()

            if self.knn_lambda_type == "trainable" and self.knn_k_type == 'trainable':
                net_outputs = self.retrieve_result_to_k_and_lambda(network_inputs)

                k_prob = net_outputs  # [B, S, R_K]

                # we add this here only to test the effect of avg prob
                if self.avg_k:
                    k_prob = torch.zeros_like(k_prob).fill_(1. / k_prob.size(-1))

                knn_lambda = 1. - k_prob[:, :, 0: 1]  # [B, S, 1]
                k_soft_prob = k_prob[:, :, 1:]
                decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                             last_hidden,
                                                                             knn_temperature,
                                                                             k_soft_prob,
                                                                             is_test=not self.retrieve_result_to_k_and_lambda.training)

            else:
                if self.knn_lambda_type == 'trainable':
                    # self.knn_distances_to_lambda[2].p = 1.0

                    knn_lambda = self.knn_distances_to_lambda(network_inputs)

                else:
                    knn_lambda = self.knn_datastore.get_lambda()

                if self.knn_k_type == "trainable":
                    # we should generate k mask
                    k_prob = self.knn_distance_to_k(network_inputs)

                    if self.knn_distance_to_k.training:
                        k_log_prob = torch.log(k_prob)
                        k_soft_one_hot = functional.gumbel_softmax(k_log_prob, tau=0.1, hard=False, dim=-1)

                    else:
                        # we get the one hot by argmax
                        _, max_idx = torch.max(k_prob, dim=-1)  # [B, S]
                        k_one_hot = torch.zeros_like(k_prob)
                        k_one_hot.scatter_(-1, max_idx.unsqueeze(-1), 1.)

                        knn_mask = torch.matmul(k_one_hot, self.knn_datastore.mask_for_distance)

                if self.knn_k_type == "trainable" and self.knn_distance_to_k.training:
                    decode_result = self.knn_datastore.calculate_select_knn_prob(knn_index, tgt_index, knn_dists,
                                                                                 last_hidden,
                                                                                 knn_temperature,
                                                                                 k_soft_one_hot)

                elif self.knn_k_type == "trainable":
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature, knn_mask)

                else:
                    decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden,
                                                                          knn_temperature)

            knn_prob = decode_result['prob']

            if self.label_count_as_feature:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index, label_counts
            else:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index

        else:
            # original situation
            return x, extra

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output. we modify this method to return prob with
        knn result
        """

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]

        # add by  , wo combine the knn prob and network prob here
        if self.use_knn_datastore:
            # x, extra, knn_probs, knn_lambda
            knn_probs = net_output[2]  # [batch, seq len, vocab size]
            knn_lambda = net_output[3]  # [batch, seq len, 1]
            network_probs = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)  # [batch, seq len, vocab size]

            if self.knn_lambda_type == "fix":
                probs = network_probs * (1 - knn_lambda) + knn_probs * knn_lambda
            else:
                probs = network_probs * (1 - knn_lambda) + knn_probs

            if log_probs:
                return torch.log(probs)
            else:
                return probs

        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

# TODO: decoder with datastore? https://github.com/zhengxxn/adaptive-knn-mt/blob/main/fairseq/models/transformer.py#L762-L844
# https://github.com/zhengxxn/adaptive-knn-mt/blob/main/fairseq/models/transformer.py#L849-L985
# https://github.com/zhengxxn/adaptive-knn-mt/blob/main/fairseq/models/transformer.py#L1196-L1238


class DeltaLMDecoderLayer(TransformerDecoderLayerBase):
    
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(TransformerDecoderLayerBase, self).__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.fc3 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc4 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.ffn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        src_lang_id = None,
        tgt_lang_id = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        ###############################################

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        ###############################################

        residual = x
        if self.normalize_before:
            x = self.ffn_layer_norm(x)

        x = self.activation_fn(self.fc3(x))
        x = self.activation_dropout_module(x)
        x = self.fc4(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.ffn_layer_norm(x)

        ###############################################

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        ###############################################
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture(
    "deltalm", "deltalm_base"
)
def base_architecture(args):
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 12
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 3072
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512


@register_model_architecture(
    "deltalm", "deltalm_large"
)
def large_architecture(args):
    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_ffn_embed_dim = 4096
    args.encoder_layers = 24
    args.encoder_attention_heads = 16
    args.encoder_normalize_before = False
    args.decoder_embed_dim = 1024
    args.decoder_ffn_embed_dim = 4096
    args.decoder_layers = 12
    args.decoder_attention_heads = 16
    args.decoder_normalize_before = False
    args.layernorm_embedding = False


@register_model_architecture(
    "deltalm_with_datastore", "deltalm_large_with_datastore"
)
def large_architecture_with_datastore(args):
    large_architecture(args)
