# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.data.multilingual.multilingual_data_manager import \
    MultilingualDatasetManager
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.transformer.adapter_helpers import \
    add_new_layers_to_pretrained
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules.adapter import get_adapter_keys, Adapter, EfficientAdapter
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerEncoderBase':
        return 'TransformerEncoder'
    else:
        return module_name


class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, hypernetwork=None):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        # ---------------------------------------------------------------------
        # Adapters
        # ---------------------------------------------------------------------
        if hasattr(cfg, "adapters") and cfg.adapters:
            self.adapters_lang = cfg.adapters_encoder_lang
            self.adapters_efficient = cfg.adapters_efficient
            self.lang_dict = MultilingualDatasetManager.create_lang_dictionary(self.cfg.langs)
            self.adapters_keys = get_adapter_keys(cfg.lang_pairs, self.adapters_lang)

            # multi-task learning: multiple (task-specific) adapters per layer
            if self.cfg.adapters_multi:
                assert self.adapters_lang is not None
            self.adapters_multi = self.cfg.adapters_multi and len(set(self.adapters_keys)) > 1

            # use separate adapters per language/task
            if self.adapters_multi:
                if self.adapters_efficient:
                    self.adapters = EfficientAdapter(
                        num_modules=len(self.adapters_keys) * cfg.encoder_layers,
                        input_size=embed_dim,
                        bottleneck_size=cfg.adapters_bottle,
                        activation_fn=cfg.adapters_activation_fn,
                        static_layernorm=cfg.adapters_static_layernorm,
                    )
                else:
                    self.adapters = nn.ModuleList([])
                    for i in range(cfg.encoder_layers):
                        self.adapters.append(
                            nn.ModuleDict({str(t): Adapter(embed_dim,
                                                           cfg.adapters_bottle,
                                                           cfg.adapters_activation_fn,
                                                           cfg.adapters_static_layernorm)
                                           for t in self.adapters_keys}))

            # use the same set of adapters for all examples
            # this is useful for single language/task finetuning
            else:
                self.adapters = nn.ModuleList([])
                for i in range(cfg.encoder_layers):
                    self.adapters.append(Adapter(embed_dim,
                                                 cfg.adapters_bottle,
                                                 cfg.adapters_activation_fn,
                                                 cfg.adapters_static_layernorm))

        else:
            self.adapters = None

        # ---------------------------------------------------------------------
        # Hyper-Adapters
        # ---------------------------------------------------------------------
#        if cfg.hyper_adapters:
#            self.hypernetwork = hypernetwork
#            self.layer2id = {f"enc-{i}": i for i in range(cfg.encoder_layers)}
#            self.id2layer = {v: k for k, v in self.layer2id.items()}
#            self.hyper_adapters_inputs = [int(x in cfg.hyper_adapters_encoder_inputs)
#                                          for x in ["src", "tgt", "layer"]]
#        else:
#            self.hypernetwork = None
#
#        # Replacement of (hyper-)adapters language -- for analysis purposes
#        if cfg.adapters_lang_swap:
#            logger.info(f"Swapping languages in adapters as follows: {cfg.adapters_lang_swap}")
#            self.adapter_lang_map = {}
#            lang_dict = MultilingualDatasetManager.create_lang_dictionary(self.cfg.langs)
#            for m in cfg.adapters_lang_swap.split(","):
#                _from, _to = m.split("-")
#                self.adapter_lang_map[lang_dict.indices[_from]] = lang_dict.indices[_to]
#        else:
#            self.adapter_lang_map = None


    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_lang_id: Optional[int] = None,
        tgt_lang_id: Optional[int] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

#        # replace the language ids used in (hyper-)adapters for analysis purposes
#        if self.adapter_lang_map is not None:
#            _src_lang_id = src_lang_id.view(-1)[0].item()
#            _tgt_lang_id = tgt_lang_id.view(-1)[0].item()
#            if _src_lang_id in self.adapter_lang_map:
#                src_lang_id = torch.zeros_like(src_lang_id).fill_(self.adapter_lang_map[_src_lang_id])
#            if _tgt_lang_id in self.adapter_lang_map:
#                tgt_lang_id = torch.zeros_like(tgt_lang_id).fill_(self.adapter_lang_map[_tgt_lang_id])
#
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings,
            src_lang_id=src_lang_id, tgt_lang_id=tgt_lang_id
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_lang_id: Optional[int] = None,
        tgt_lang_id: Optional[int] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            #  <adapters>
            if self.adapters is not None:
                if self.adapters_multi:
                    # we expect all src-tgt samples contain the same task/language
                    if self.adapters_lang == "src":
                        key = self.lang_dict.symbols[src_lang_id[0]]
                    elif self.adapters_lang == "tgt":
                        key = self.lang_dict.symbols[tgt_lang_id[0]]
                    elif self.adapters_lang == "pair":
                        src_id = self.lang_dict.symbols[src_lang_id[0]]
                        tgt_id = self.lang_dict.symbols[tgt_lang_id[0]]
                        key = f"{src_id}-{tgt_id}"

                    if self.adapters_efficient:
                        key_id = self.adapters_keys.index(key)
                        adapter_id = len(self.adapters_keys) * idx + key_id
                        x = self.adapters(x, adapter_id)
                    else:
                        x = self.adapters[idx][key](x)
                else:
                    x = self.adapters[idx](x)
            #  </adapters>

            #  <hyper-adapters>
            #elif self.hypernetwork is not None:
            #    # note that, src_lang_id and tgt_lang_id start from 1, and
            #    # we assume all src-tgt samples contain the same task/language
            #    _src_lang_id = src_lang_id[0].squeeze() - 1
            #    _tgt_lang_id = tgt_lang_id[0].squeeze() - 1
            #    layer_id = torch.tensor(self.layer2id[f"enc-{idx}"], device=x.device)
            #    x = self.hypernetwork(x, _src_lang_id, _tgt_lang_id, layer_id,
            #                          self.hyper_adapters_inputs)
            #  </hyper-adapters>

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )
