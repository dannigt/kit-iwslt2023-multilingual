# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerEncoderBase,
    TransformerDecoderBase,
    TransformerConfig,
)
from torch import Tensor

logger = logging.getLogger(__name__)

class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

        if hasattr(cfg, "adapters") and cfg.adapters: # or cfg.hyper_adapters:
            assert cfg.one_dataset_per_batch, "(hyper-)adapters require `--one-dataset-per-batch`"
            assert cfg.enable_lang_ids, "(hyper-)adapters require `--enable-lang-ids`"

            if cfg.freeze_pretrained:
                for n, p in self.named_parameters():
                    if not hasattr(p, "hyper") and not hasattr(p, "adapter"):
                        logger.info(f"Freezing (pretrained) parameter '{n}'")
                        p.requires_grad = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

        # ------------------------------------------------------------------------------------------------
        # args for adapter layers
        # ------------------------------------------------------------------------------------------------
        parser.add_argument('--adapters', default=False, action='store_true',
                            help="Add an adapter layer after each Transformer layer.")
        parser.add_argument('--adapters-multi', default=False, action='store_true',
                            help="Use multiple different adapters, one per task (e.g., language(-pair)).")
        parser.add_argument('--adapters-bottle', type=int, metavar='D',
                            help="The dimensionality of the adapter bottleneck representation.")
        parser.add_argument('--adapters-activation-fn', type=str, default="relu",
                            choices=utils.get_available_activation_fns(),
                            help="Activation function for the adapters bottleneck.")
        parser.add_argument('--adapters-static-layernorm',  default=False, action='store_true',
                            help="Use LayerNorm without trainable parameters in adapter layers.")
        parser.add_argument('--adapters-encoder-lang', default="src", type=str,
                            choices=['src', 'tgt', 'pair'],
                            help='"The key to use for indexing the encoder language adapters."')
        parser.add_argument('--adapters-decoder-lang', default="tgt", type=str,
                            choices=['src', 'tgt', 'pair'],
                            help="The key to use for indexing the decoder language adapters.")
        parser.add_argument('--adapters-efficient', default=False, action='store_true',
                            help="Use a single weight matrix for all adapter layers. "
                                 "Improves distributed training efficiency.")

        # ------------------------------------------------------------------------------------------------
        # args for hyper-adapter layers
        # ------------------------------------------------------------------------------------------------
        parser.add_argument('--hyper-adapters', default=False, action='store_true',
                            help="Add a hyper-adapter layer after each Transformer layer.")
        parser.add_argument('--hyper-adapters-bottle', type=int, metavar='D',
                            help="The dimensionality of the generated adapter bottleneck representation.")
        parser.add_argument('--hyper-adapters-hidden-dim', type=int, metavar='D',
                            help="The dimensionality of the hyper-network hidden representations.")
        parser.add_argument('--hyper-adapters-hidden-layers', type=int, metavar='D', default=1,
                            help="The number of hidden layers in the hyper-network.")
        parser.add_argument('--hyper-adapters-lang-embed-dim', type=int, metavar='D',
                            help="The dimensionality of the language embeddings of the hyper-network.")
        parser.add_argument('--hyper-adapters-layer-embed-dim', type=int, metavar='D',
                            help="The dimensionality of the layer embeddings of the hyper-network.")

        parser.add_argument('--hyper-adapters-dropout', type=float, metavar='D', default=0.0,
                            help="Dropout used in the hyper-network.")
        parser.add_argument('--hyper-adapters-activation-fn', type=str, default="relu",
                            choices=utils.get_available_activation_fns(),
                            help="Activation function for the hyper-network "
                                 "and the generated adapters bottleneck.")
        parser.add_argument('--hyper-adapters-lang-embed-tied', default=False, action='store_true',
                            help="Use a share embedding matrix for the source and target language embeddings.")
        parser.add_argument('--hyper-adapters-layernorm-input', default=False, action='store_true',
                            help="Apply layer normalization to the hyper-network input (lang+layer embeddings).")
        parser.add_argument('--hyper-adapters-layernorm-output', default=False, action='store_true',
                            help="Apply layer normalization to the hyper-network output "
                                 "(before weight generation projections).")
        parser.add_argument('--hyper-adapters-generate-layernorm', default=False, action='store_true',
                            help="Generate from the hyper-network the LayerNorm "
                                 "parameters for each generated adapter layer.")
        parser.add_argument('--hyper-adapters-init', default="fairseq", type=str, choices=['default', 'hyper'],
                            help='Initialization method for the weights of the hyper-network.')
        parser.add_argument('--hyper-adapters-encoder-inputs', default="src,tgt,layer", type=str,
                            help="Which sources of information to use as input to the "
                                 "hyper-network when generating the encoder hyper-adapters.")
        parser.add_argument('--hyper-adapters-decoder-inputs', default="src,tgt,layer", type=str,
                            help="Which sources of information to use as input to the "
                                 "hyper-network when generating the decoder hyper-adapters.")
        parser.add_argument('--hyper-adapters-no-rescale', default=False, action='store_true',
                            help="Disable the weight rescaling in the hyper-network.")

        # args for finetuning
        parser.add_argument('--freeze-pretrained', default=False, action='store_true',
                            help="Freeze the parameters of the main network. "
                                 "Applicable to experiments for (hyper-)adapters finetuning.")

        # args for analysis
        parser.add_argument('--network-inspection', default=False, action='store_true',
                            help="Log and visualize the weights and activations of selected layers.")
        parser.add_argument('--adapters-lang-swap', type=str, default=None,
                            help="Swap the (hyper-)adapter of one language with another. "
                                 "Expects a comma-separated string "
                                 "(e.g., `--adapter-lang-swap af_ZA-nl_XX,pt_XX-es_XX,gl_ES-pt_XX,uk_UA-ru_RU`)")

        parser.add_argument('--debugging', default=False, action='store_true',
                            help="debugging")

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing

        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens) #, hypernetwork)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens) #, hypernetwork)

        if not cfg.share_all_embeddings:
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=cfg.min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=cfg.min_params_to_wrap)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens, hypernetwork=None):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens, hypernetwork=hypernetwork)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens, hypernetwork=None):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
            hypernetwork=hypernetwork
        )
    @classmethod
    def build_hypernetwork(cls, cfg, task):
        hypernetwork = None
        return hypernetwork

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lang_id: Optional[int] = None,
        tgt_lang_id: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
