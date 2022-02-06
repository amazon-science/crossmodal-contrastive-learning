from collections import OrderedDict
from easydict import EasyDict
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from utils_collection.general import truncated_normal_fill
from utils_collection.config_tools import TransformerConfig

from functools import partial

from models.vit import VisionTransformer, ResEncoderBlockCTX, AttentionCTX
from models.vit_tools import trunc_normal_
from reformer_pytorch import Reformer, LSHAttention, ReformerLM


__all__ = [
    'mme_tiny', 'mme_small', 'mme_base', 'mme_large'
    ]


def mme_tiny_reformer(cfg: EasyDict, feature_dim: int, max_frames: int = 40):
    # print(max_h, "max_h")
    # print([max_h, feature_dim], cfg.input_fc_dim, "00"*10)
    # model_vid_local = ExpertTransformer(inp_size=[max_h, cfg.input_fc_dim], patch_size=patch_local, in_chans_clip=2, in_chans_ctx=5, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=1, num_heads=8, mlp_ratio=1, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim)


    model_local = Reformer(
        dim = 1024,
        depth = 1,
        max_seq_len = max_frames,
        heads = 4,
        lsh_dropout = 0.1,
        causal = True,
        bucket_size = 4,
    )

    model_context = Reformer(
        dim = 1024,
        depth = 1,
        max_seq_len = max_frames,
        heads = 4,
        lsh_dropout = 0.1,
        causal = True,
        bucket_size = 5,
    )

    # x = torch.randn(1, 8192, 512).cuda()
    return model_local, model_context


def mme_tiny(cfg: EasyDict, feature_dim: int, max_frames: int = 40, patching_on: bool = False):
    patch_local = [4, 16]
    max_h = max_frames // 5
    # print(max_h, "max_h")
    # print([max_h, feature_dim], cfg.input_fc_dim, "00"*10)
    patching_on = False #currently not working! error during evaluation because of reshape problem
    vir_flag = True
    if patching_on:

        model_vid_local = ExpertTransformer(inp_size=[max_h, cfg.input_fc_dim], patching_on=patching_on, patch_size=patch_local, in_chans=2, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=2, num_heads=8, mlp_ratio=1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim, pooler_config=cfg, vir_flag=vir_flag)

        model_vid_context = ExpertTransformer(inp_size=[max_h, cfg.input_fc_dim], patching_on=patching_on, patch_size=patch_local, in_chans=5, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=2, num_heads=8, mlp_ratio=1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim, pooler_config=cfg, vir_flag=vir_flag)
    else:

        model_vid_local = ExpertTransformer(inp_size=[8, cfg.input_fc_dim], patching_on=patching_on, patch_size=patch_local, in_chans=2, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=2, num_heads=8, mlp_ratio=1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim, pooler_config=cfg, vir_flag=vir_flag)

        model_vid_context = ExpertTransformer(inp_size=[20, cfg.input_fc_dim], patching_on=patching_on, patch_size=patch_local, in_chans=5, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=2, num_heads=8, mlp_ratio=1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim, pooler_config=cfg, vir_flag=vir_flag)
     
    cfg.pooler_config.name = "avg"
    model_vid_global = ExpertTransformer(inp_size=[5,5], patching_on=False, patch_size=patch_local, in_chans=2, embed_dim=cfg.input_fc_dim, num_classes=cfg.output_dim, depth=1, num_heads=8, mlp_ratio=1, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=True, input_fc_dim=cfg.input_fc_dim, feature_dim=cfg.input_fc_dim, pooler_config=cfg, vir_flag=vir_flag)

    # model_vid_global = ExpertTransformer(inp_size=[5, cfg.output_dim], patch_size=[1, 16],  in_chans_clip=1, in_chans_ctx=5, embed_dim=192, num_classes=cfg.output_dim, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, use_input_fc=False, input_fc_dim=cfg.input_fc_dim, feature_dim=feature_dim)

    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model_vid_local, model_vid_context, model_vid_global

class TransformerReformLM(nn.Module):
    ##NOTE: to use this class we need to quantize visual features and also use language word embeddings not features!
    def __init__(self, cfg: EasyDict, feature_dim: int, max_frames: int, bucket_size: int, subspace_size: int):
        super().__init__()
        
        self.input_fc = None
        self.subspace_size = subspace_size
        self.bucket_size = bucket_size
        input_dim = feature_dim
        self.input_norm = LayerNormalization(input_dim)

        # if cfg.use_input_fc:
        #     self.input_fc = nn.Sequential(
        #         nn.Linear(feature_dim, cfg.input_fc_dim), nn.GELU())
        #     input_dim = cfg.input_fc_dim

        self.embedding = PositionalEncoding(input_dim,
                                            cfg.selfatn.dropout,
                                            max_len=1000)

        self.block = ReformerLM(
            num_tokens= feature_dim,
            dim = cfg.input_fc_dim//subspace_size,
            depth = cfg.selfatn.num_layers,
            max_seq_len = max_frames,
            heads = 8,
            lsh_dropout = cfg.selfatn.dropout,
            ff_dropout = cfg.selfatn.dropout,
            post_attn_dropout = cfg.selfatn.dropout,
            layer_dropout = cfg.selfatn.dropout,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
            causal = True,        # auto-regressive or not
            bucket_size = bucket_size,     # average size of qk per bucket, 64 was recommended in paper
            n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
            emb_dim = cfg.input_fc_dim,        # embedding factorization for further memory savings
            dim_head = 64,        # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
            ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
            attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
            num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
            twin_attention = False, # both branches of the reversible network will be attention
            full_attn_thres = 1024, # use full attention if context length is less than set value
            reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
            use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
            use_rezero = False,      # remove normalization and use rezero from 'ReZero is All You Need'
            one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
            weight_tie = False,           # tie parameters of each layer for no memory per additional depth
            weight_tie_embedding = False, # use token embedding for projection of output, some papers report better results
            n_local_attn_heads = 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
            pkm_layers = (4,7),           # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
            pkm_num_keys = 128,           # defaults to 128, but can be increased to 256 or 512 as memory allows
            use_full_attn = False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
        ).cuda()
        # self.block = Reformer(dim=input_dim//subspace_size, depth=cfg.selfatn.num_layers,
        #                                          max_seq_len=max_frames,
        #                                          heads=cfg.selfatn.num_heads,
        #                                          bucket_size=bucket_size,
        #                                          lsh_dropout=cfg.selfatn.dropout)

        self.use_context = cfg.use_context
        if self.use_context:
            print("Nothing")
            # self.block_context = Reformer(dim=input_dim//subspace_size, depth=cfg.selfatn.num_layers,
            #                                      max_seq_len=max_frames,
            #                                      heads=cfg.selfatn.num_heads,
            #                                      bucket_size=bucket_size,
            #                                      lsh_dropout=cfg.selfatn.dropout)

        self.pooler = parse_pooler(input_dim, cfg)

        init_network(self, init_std=0.01)

    def forward(self, features, ctx_h=None, ctx_flag=False):
        # print(features.shape)
        # features = self.input_norm(features)
        # if self.input_fc is not None:
        #     features = self.input_fc(features)
        features = self.embedding(features)
        # print(features.shape)
        features = features.reshape(features.shape[0], features.shape[1]*self.subspace_size, features.shape[-1]//self.subspace_size)
        print(features.shape)
        x = torch.randint(0, 20000, (1, 8192)).long().cuda()
        print(x.shape, "-0-"*10)
        features = self.block(features)
        print(features.shape)
        features = features.reshape(features.shape[0], features.shape[1]//self.subspace_size, -1)

        pooled = self.pooler(features) #, None, None)

        add_after_pool = None
        # if ctx_flag:
        #     features = self.block_context(features)

        if False:
            ctx_h = ctx_h.unsqueeze(1)
            ctx = self.tf_context(ctx_h, features, features, mask)
            add_after_pool = ctx.squeeze(1)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        return pooled

class TransformerReform(nn.Module):
    def __init__(self, cfg: EasyDict, feature_dim: int, max_frames: int, bucket_size: int, subspace_size: int):
        super().__init__()
        
        self.input_fc = None
        self.subspace_size = subspace_size
        self.bucket_size = bucket_size
        input_dim = feature_dim
        self.input_norm = LayerNormalization(input_dim)

        if cfg.use_input_fc:
            self.input_fc = nn.Sequential(
                nn.Linear(feature_dim, cfg.input_fc_dim), nn.GELU())
            input_dim = cfg.input_fc_dim

        self.embedding = PositionalEncoding(input_dim,
                                            cfg.selfatn.dropout,
                                            max_len=1000)

        self.block = Reformer(dim=input_dim//subspace_size, depth=cfg.selfatn.num_layers,
                                                 max_seq_len=max_frames,
                                                 heads=cfg.selfatn.num_heads,
                                                 bucket_size=bucket_size,
                                                 lsh_dropout=cfg.selfatn.dropout)

        self.use_context = cfg.use_context
        if self.use_context:
            self.ctx_attn = LSHAttention(
                                bucket_size = 8,
                                n_hashes = 4,
                                causal = True
                            )

        self.pooler = parse_pooler(input_dim, cfg)

        init_network(self, init_std=0.01)

    def forward(self, features, ctx_h=None, ctx_flag=False):
        # print(features.shape, "input shape Reformer")
        # features = self.input_norm(features)
        if self.input_fc is not None:
            features = self.input_fc(features)
        features = self.embedding(features)
        print(features.shape)
        print(features.shape[0], features.shape[1]*self.subspace_size, features.shape[-1]//self.subspace_size)
        features = features.reshape(features.shape[0], features.shape[1]*self.subspace_size, features.shape[-1]//self.subspace_size)
        print(features.shape, "after reshaping")
        features = self.block(features)
        features = features.reshape(features.shape[0], features.shape[1]//self.subspace_size, -1)

        pooled = self.pooler(features) #, None, None)

        add_after_pool = None
        if ctx_flag:
            # print(ctx_h.shape, features.shape, "ctx"*5)
            ctx = self.ctx_attn(ctx_h, features)
            add_after_pool = ctx.squeeze(1)[:,0,:]

        # if False:
        #     ctx_h = ctx_h.unsqueeze(1)
        #     ctx = self.tf_context(ctx_h, features, features, mask)
        #     add_after_pool = ctx.squeeze(1)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        return pooled

class ExpertTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrupt_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.embedding = PositionalEncoding(self.feature_dim,
                                            self.drop_path_rate,
                                            max_len=1000)
        if self.patching_on:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.inp_size[0] + 2, self.embed_dim))
           

            
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.input_fc = nn.Linear(self.feature_dim, self.input_fc_dim) if self.use_input_fc else nn.Identity()
        # print(self.feature_dim, self.input_fc_dim, self.embed_dim, self.num_classes, "===>0")
        trunc_normal_(self.corrupt_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.pooler = parse_pooler(self.input_fc_dim, self.pooler_config)

        self.ctx_type = "realformer"
        ctx_depth = 2
        if self.ctx_type =="lsha":
            self.ctx_attn = LSHAttention(
                bucket_size = 8,
                n_hashes = 4,
                causal = True
            )
        elif self.ctx_type =="atn":
            self.ctx_attn = nn.Sequential(*[AttentionCTX(input_dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.05, proj_drop=0.05) for _ in range(ctx_depth)])
        elif self.ctx_type =="realformer":
            
            self.ctx_attn = nn.Sequential(*[ResEncoderBlockCTX(emb_s = self.embed_dim//8, head_cnt = 8, dp1 = 0.05, dp2 = 0.1) for _ in range(ctx_depth)])

    def forward_ctx_LSH(self, x, ctx_h):
        x = x.reshape(-1, 16, x.shape[2]//16)
        ctx_h = ctx_h.reshape(-1, 16, ctx_h.shape[1]//16)
        ctx, _, _ = self.ctx_attn(ctx_h, x)
        ctx = ctx.reshape(ctx.shape[0], ctx.shape[1]*ctx.shape[2])
        return ctx

    def forward_ctx_atn(self, x, ctx_h):
        for atn in self.ctx_attn:
            x = atn(x, ctx_h)
        return x

    def forward_ctx_realformer(self, x, ctx_h):
        prev = None
        for resencoder in self.ctx_attn:
            x, prev = resencoder(x, ctx_h, prev = prev)
        return x

    def forward_features(self, x, ctx_flag):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        # x shape: [batch, 196 (depends on patching), 192(depends on the model)]
        # ctx_flag: Video context or Clip
        B = x.shape[0]
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape, self.patching_on, "====>")
        # print((x.shape[0], self.in_chans, -1, x.shape[-1]))
        if self.patching_on:
            x = x.reshape((x.shape[0], self.in_chans, -1, x.shape[-1]))
                # print(x.shape, self.in_chans_clip, "===>in_chans_clip")


        # print(x.shape)
        # x = torch.zeros(1, 6, 1, 512).cuda()
        # print(self.use_input_fc, "self.use_input_fc")
        if self.use_input_fc:
            x = self.input_fc(x)
            # print(x.shape,"===>Input FC"*10)
        if self.patching_on:
            x = self.patch_embed(x, ctx_flag)

        # print(x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks [Batch * 1 * 192]
        corrupt_token = self.corrupt_token.expand(B, -1, -1) # [Batch , 1, 192]
        # print(cls_tokens.shape, corrupt_token.shape, x.shape)
        x = torch.cat((cls_tokens, corrupt_token, x), dim=1)
        # print(x.shape, self.pos_embed.shape, "--00--"*10)
        if self.patching_on:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if self.vir_flag:
            prev = None
            for blk in self.blocks:
                x, prev = blk(x, prev = prev)
        else:
            for blk in self.blocks:
                x = blk(x)

        
        x = self.norm(x)
        return x #[:, 0], x[:, 1]

    def forward(self, x, ctx_h=None, ctx_flag=False):
        # print(x.shape)
        x = self.forward_features(x, ctx_flag)
        # x = self.head(x)
        # print(x.shape)
        # x_dist = self.head_dist(x_dist)
        # if self.training:
        pooled = self.pooler(x)
        add_after_pool = None
        if ctx_flag:
            # print(x.shape, ctx_h.shape, "===>")
            if self.ctx_type == "lsha":
                add_after_pool = self.forward_ctx_LSH(x, ctx_h)
            elif self.ctx_type == "atn":
                # add_after_pool,_ = torch.max(self.ctx_attn(x, ctx_h), dim=1)
                add_after_pool = torch.mean(self.forward_ctx_atn(features, hidden_state), dim=1)
            elif self.ctx_type == "realformer":
                # add_after_pool = self.forward_ctx_realformer(x, ctx_h)[:,0,:]
                add_after_pool,_ = torch.max(self.forward_ctx_realformer(x, ctx_h), dim=1)
                # add_after_pool = self.forward_ctx_atn(x, ctx_h)[:,0,:]


        if add_after_pool is not None:
            # print(pooled.shape, add_after_pool.shape)
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
            # print("pooled shape: ", pooled.shape)
        # print(pooled.shape, "final return shape Reformer")
        return pooled#, x_dist
        # else:
        #     # during inference, return the average of both classifier predictions
        #     return (x + x_dist) / 2


class MultiModalTransformer:
    def __init__(self, cfg: TransformerConfig, use_cuda: bool, use_multi_gpu: bool):
        self.use_cuda = use_cuda
        self.use_multi_gpu = use_multi_gpu
        self.model_list = []

        self.cfg: TransformerConfig = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")

       
        self.video_context = self.video_local = TransformerReform(cfg.model_cfgs['video_local'],
                                            cfg.dataset.vid_feat_dim, cfg.dataset.max_frames, bucket_size=8, subspace_size=32)
        # print(cfg.dataset.max_frames, "maxframes===")
        # self.video_local, self.video_context, self.video_global = mme_tiny(cfg.model_cfgs['video_local'], cfg.dataset.vid_feat_dim, cfg.dataset.max_frames)                                  
        
        self.video_local = self._to_device_fn(self.video_local)
        self.video_context = self._to_device_fn(self.video_context)

        self.video_global = Transformer(cfg.model_cfgs['video_global'],
                                               cfg.model_cfgs['video_global'].output_dim)
        # self.video_global = TransformerReform(cfg.model_cfgs['video_global'],
        #                                     cfg.model_cfgs['video_local'].output_dim, cfg.dataset.max_frames, bucket_size=8, subspace_size=32)
        self.video_global = self._to_device_fn(self.video_global)
        self.text_local = Transformer(cfg.model_cfgs['text_local'],
                                           cfg.dataset.text_feat_dim)
        self.text_local = self._to_device_fn(self.text_local)
        self.text_global = Transformer(cfg.model_cfgs['text_global'],
                                              cfg.model_cfgs['text_global'].output_dim)
        self.text_global = self._to_device_fn(self.text_global)
        self.model_list = [
            self.video_local, self.video_global,
            self.text_local, self.text_global
        ]

    def encode_video(self, vid_frames, vid_frames_mask, vid_frames_len,
                     clip_num, clip_frames, clip_frames_len, clip_frames_mask):
        # compute video context
        # print(vid_frames.shape, "vid_frames")
        ctx_flag = False
        vid_context = self.video_context(vid_frames, ctx_flag=ctx_flag) #, ctx_flag)
        # vid_context = vid_context_tmp[:, 0, :]
        # vid_context = self.video_local(vid_frames, vid_frames_mask,
        #                                     vid_frames_len, None)
        # print(vid_context.shape, "vid context")
        if self.cfg.model_cfgs['video_global'].use_context:
            if self.cfg.model_cfgs['video_global'].name == "rnn":
                vid_context_hidden = vid_context.unsqueeze(0)
                vid_context_hidden = vid_context_hidden.repeat(
                    self.cfg.model_cfgs['video_global'].crossatn_config.num_layers, 1, 1)
            elif self.cfg.model_cfgs['video_global'].name == "transformer":
                vid_context_hidden = vid_context
            else:
                raise NotImplementedError
        else:
            vid_context_hidden = None

        # compute clip embedding
        # clip_emb = self.video_local(clip_frames, clip_frames_mask,
        #                                  clip_frames_len, None)
        # print(clip_frames.shape, "clip_frames")
        ctx_flag = False
        clip_emb = self.video_local(clip_frames, ctx_flag=ctx_flag) #, ctx_flag)
        # print(clip_emb.shape, "clip emb")
        # clip_emb = clip_emb_tmp[:, 0, :]
 

        batch_size = len(clip_num)
        max_clip_len = torch.max(clip_num)
        clip_feat_dim = self.cfg.model_cfgs['video_local'].output_dim
        clip_emb_reshape = torch.zeros(
            (batch_size, max_clip_len, clip_feat_dim))
        clip_emb_mask = torch.zeros((batch_size, max_clip_len))
        clip_emb_lens = torch.zeros((batch_size, ))
        if self.use_cuda:
            clip_emb_reshape = clip_emb_reshape.cuda(non_blocking=True)
            clip_emb_mask = clip_emb_mask.cuda(non_blocking=True)
            clip_emb_lens = clip_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, clip_len in enumerate(clip_num):
            clip_emb_reshape[batch, :clip_len, :] =\
                clip_emb[pointer:pointer + clip_len, :]
            clip_emb_mask[batch, :clip_len] = 1
            clip_emb_lens[batch] = clip_len
            pointer += clip_len

        # compute video embedding
        # print(clip_emb_reshape.shape, "clip_emb_reshape")
        ctx_flag = True
        # vid_emb = self.video_global(clip_emb_reshape, vid_context_hidden, ctx_flag)
        vid_emb = self.video_global(clip_emb_reshape, clip_emb_mask,
                                           clip_num, vid_context_hidden)
        # vid_emb = self.video_global(clip_emb_reshape)
        # print(vid_emb.shape, "===vid_emb_global"*5)

        #TODO: convert the return to an object class or maybe dictionary
        return (vid_emb, clip_emb, vid_context, clip_emb_reshape,
                clip_emb_mask, clip_emb_lens)

    def encode_paragraph(self, paragraph_caption_vectors, paragraph_caption_mask, paragraph_caption_len,
                         sentence_num, sentence_caption_vectors, sentence_caption_mask,
                         sentence_caption_len):
        # compute paragraph context
        paragraph_context = self.text_local(paragraph_caption_vectors, paragraph_caption_mask,
                                           paragraph_caption_len, None)

        if self.cfg.model_cfgs['text_global'].use_context:
            if self.cfg.model_cfgs['text_global'].name == "rnn":
                paragraph_gru_hidden = paragraph_context.unsqueeze(0)
                paragraph_gru_hidden = paragraph_gru_hidden.repeat(
                    self.cfg.model_cfgs['text_global'].crossatn_config.num_layers, 1, 1)
            elif self.cfg.model_cfgs['text_global'].name == "transformer":
                paragraph_gru_hidden = paragraph_context
            else:
                raise NotImplementedError
        else:
            paragraph_gru_hidden = None

        # compute sentence embedding
        sentence_emb = self.text_local(sentence_caption_vectors, sentence_caption_mask,
                                        sentence_caption_len, None)
        # print(sentence_emb.shape, "sentence shape")
        batch_size = len(sentence_num)
        sentence_feat_dim = self.cfg.model_cfgs['text_local'].output_dim
        max_sentence_len = torch.max(sentence_num)
        sentence_emb_reshape = torch.zeros(
            (batch_size, max_sentence_len, sentence_feat_dim))
        sentence_emb_mask = torch.zeros((batch_size, max_sentence_len))
        sentence_emb_lens = torch.zeros((batch_size, ))
        if self.use_cuda:
            sentence_emb_reshape = sentence_emb_reshape.cuda(non_blocking=True)
            sentence_emb_mask = sentence_emb_mask.cuda(non_blocking=True)
            sentence_emb_lens = sentence_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, sentence_len in enumerate(sentence_num):
            sentence_emb_reshape[batch, :sentence_len, :] =\
                sentence_emb[pointer:pointer + sentence_len, :]
            sentence_emb_mask[batch, :sentence_len] = 1
            sentence_emb_lens[batch] = sentence_len
            pointer += sentence_len

        # compute paragraph embedding
        paragraph_emb = self.text_global(sentence_emb_reshape, sentence_emb_mask,
                                          sentence_num, paragraph_gru_hidden)
        # print(paragraph_emb.shape, "paragraph shape")

        #TODO: convert the return to a object class or dictionary
        return (paragraph_emb, sentence_emb, paragraph_context, sentence_emb_reshape,
                sentence_emb_mask, sentence_emb_lens)

    def eval(self):
        for model in self.model_list:
            model.eval()
        torch.set_grad_enabled(False)

    def train(self):
        for model in self.model_list:
            model.train()
        torch.set_grad_enabled(True)

    def _to_device_fn(self, model):
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(self.device)
        return model

    def get_all_params(self) -> Tuple[Any, Any, Any]:
        """
        Since there are multiple networks used by this trainer, this
        function can be used to get all the parameters at once.


        Returns:
            params, param_names, params_flat
        """
        # loop models and collect parameters
        params, param_names, params_flat = [], [], []
        for model in self.model_list:
            _params, _param_names, _params_flat = self.get_params_opt_simple(model)
            params.extend(_params)
            param_names.extend(_param_names)
            params_flat.extend(_params_flat)
        return params, param_names, params_flat

    def get_params_opt_simple(self, model: nn.Module) -> (
            Tuple[List[Dict[str, Any]], List[str], List[torch.Tensor]]):
        """
        Args:
            model: Model to get the parameters from.

        Returns:
            Tuple of:
                List of:
                    Dict of:
                        'params': The parameter
                        'decay_mult': Multiply weight decay with this factor
                        'lr_mult': Multiplay learning rate with this factor
                List of:
                    parameter names
                List of:
                    parameters
        """
        params_dict: Dict[str, torch.Tensor] = dict(model.named_parameters())
        params, param_names, params_flat = [], [], []
        # print(cfg.training.representation)
        for key, value in params_dict.items():
            decay_mult = 1.0
            if self.cfg.optimizer.weight_decay_for_bias and 'bias' in key:
                decay_mult = 0.0
            params += [{
                'params': value,
                'decay_mult': decay_mult,
                'lr_mult': 1.0
            }]
            param_names += [key]
            params_flat += [value]

        return params, param_names, params_flat

    def get_params(self):
        params = []
        for model in self.model_list:
            params_dict = OrderedDict(model.named_parameters())
            _params = []
            for key, value in params_dict.items():
                _params += [{'params': value}]
            params.extend(_params)
        return params

    def load_checkpoint(self, ckpt: str):
        state = torch.load(str(ckpt))
        for i, model in enumerate(self.model_list):
            state_dict = state[i]
            if self.use_multi_gpu:
                newer_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert not key.startswith("module.")
                    new_key = "module." + key
                    newer_state_dict[new_key] = val
                model.load_state_dict(newer_state_dict)
            else:
                model.load_state_dict(state_dict)
            i += 1 # we do this intentionally
            
    def get_model_state(self):
        model_states = []
        for m in self.model_list:
            state_dict = m.state_dict()
            if self.use_multi_gpu:
                new_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert key.startswith("module.")
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                model_states.append(new_state_dict)
            else:
                model_states.append(state_dict)
        return model_states

    def save_checkpoint(self, ckpt: str):
        model_states = []
        for m in self.model_list:
            state_dict = m.state_dict()
            if self.use_multi_gpu:
                new_state_dict = OrderedDict()
                for key, val in state_dict.items():
                    assert key.startswith("module.")
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                model_states.append(new_state_dict)
            else:
                model_states.append(state_dict)
        torch.save(model_states, str(ckpt))


class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features_count),
                                 requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(features_count),
                                 requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gain * (x - mean) / (std + self.epsilon) + self.bias



def parse_pooler(input_dim, cfg: EasyDict) -> nn.Module:
    # TODO: This part can be improved by finding smarter pooling methods.

    if cfg.pooler_config.name == "atn":
        return AtnPool(input_dim, cfg.pooler_config.hidden_dim, cfg.pooler_config.num_heads,
                         cfg.pooler_config.dropout)
    elif cfg.pooler_config.name == "avg":
        return AvgPool()
    elif cfg.pooler_config.name == "max":
        return MaxPool()
    else:
        raise ValueError(f"unknown pooler {cfg.pooler_config.name}")


class Transformer(nn.Module):
    def __init__(self, cfg: EasyDict, feature_dim: int):
        super().__init__()
        
        self.input_norm = LayerNormalization(feature_dim)
        self.input_fc = None
        input_dim = feature_dim
        if cfg.use_input_fc:
            self.input_fc = nn.Sequential(
                nn.Linear(feature_dim, cfg.input_fc_dim), nn.GELU())
            input_dim = cfg.input_fc_dim
        self.embedding = PositionalEncoding(input_dim,
                                            cfg.selfatn.dropout,
                                            max_len=1000)

        self.tf = TransformerEncoder(cfg.selfatn.num_layers, input_dim, cfg.selfatn.num_heads,
                                     input_dim, cfg.selfatn.dropout)

        self.use_context = cfg.use_context
        if self.use_context:
            self.tf_context = TransformerEncoder(cfg.selfatn.num_layers,
                                                 input_dim,
                                                 cfg.selfatn.num_heads,
                                                 input_dim, cfg.selfatn.dropout)

        self.pooler = parse_pooler(input_dim, cfg)
        self.ctx_attn = LSHAttention(
            bucket_size = 4,
            n_hashes = 4,
            causal = True
        )

        init_network(self, init_std=0.01)

    def forward(self, features, mask, lengths, hidden_state):
        features = self.input_norm(features)
        if self.input_fc is not None:
            features = self.input_fc(features)
        features = self.embedding(features)
        features = self.tf(features, features, features, mask)
        add_after_pool = None

        pooled = self.pooler(features) #mask, lengths)
        # print(features.shape, pooled.shape, "Shape language input")
        if self.use_context:
            # hidden_state = hidden_state.unsqueeze(1)

            # print(features.shape, hidden_state.shape)
            features = features.reshape(-1, 16, features.shape[2]//16)
            hidden_state = hidden_state.reshape(-1, 16, hidden_state.shape[1]//16)
            # print(features.shape, hidden_state.shape, "=00==>")
            ctx, _, _ = self.ctx_attn(hidden_state, features)
            # ctx = self.tf_context(hidden_state, features, features, mask)
            ctx = ctx.reshape(ctx.shape[0], ctx.shape[1]*ctx.shape[2])

            add_after_pool = ctx.squeeze(1)
            # print(add_after_pool.shape, "add after pool shape")
        if add_after_pool is not None:
            # print(pooled.shape, add_after_pool.shape)
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
        # print(pooled.shape, "final return shape language")
        return pooled


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000**(2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        # print(x.shape, self.pe.shape)
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, model_dim, heads_count, fc_dim, dropout_prob):
        super().__init__()
        self.model_dim = model_dim
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, heads_count, fc_dim, dropout_prob)
            for _ in range(layers_count)
        ])

    def forward(self, query, key, value, mask):
        # print(query.shape, "====>")
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, heads_count, fc_dim, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, model_dim, dropout_prob), model_dim)
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(fc_dim, model_dim, dropout_prob), model_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, model_dim):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(model_dim)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, model_dim, dropout_prob):
        super().__init__()
        assert model_dim % heads_count == 0,\
            f"model dim {model_dim} not divisible by {heads_count} heads"
        self.d_head = model_dim // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.key_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.value_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.final_projection = nn.Linear(model_dim, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, model_dim = query.size()
        d_head = model_dim // self.heads_count
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, model_dim = key_projected.size()
        batch_size, value_len, model_dim = value_projected.size()
        query_heads = query_projected.view(batch_size, query_len,
                                           self.heads_count,
                                           d_head).transpose(1, 2)
        key_heads = key_projected.view(batch_size, key_len, self.heads_count,
                                       d_head).transpose(1, 2)
        value_heads = value_projected.view(batch_size, value_len,
                                           self.heads_count,
                                           d_head).transpose(1, 2)
        attention_weights = self.scaled_dot_product(query_heads, key_heads)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(
                mask_expanded, -1e18)
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)
        context_heads = torch.matmul(attention_dropped, value_heads)
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(batch_size, query_len, model_dim)
        final_output = self.final_projection(context)
        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)
        dot_product = torch.matmul(query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, fc_dim, model_dim, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(model_dim, fc_dim),
                                          nn.Dropout(dropout_prob), nn.GELU(),
                                          nn.Linear(fc_dim, model_dim),
                                          nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)

class AvgPool(nn.Module):
    def forward(self, features):
        result = torch.mean(features, dim=1)
        # result = result_sum / len_div
        return result

class MaxPool(nn.Module):
    def forward(self, features):
        result_max, _= torch.max(features, dim=1)
        return result_max

class AtnPool(nn.Module):
    def __init__(self, d_input, d_attn, n_heads, dropout_prob):
        super().__init__()
        self.d_head = d_attn // n_heads
        self.d_head_output = d_input // n_heads
        self.num_heads = n_heads

        def _init(tensor_):
            tensor_.data = (truncated_normal_fill(tensor_.data.shape,
                                                        std=0.01))

        w1_head = torch.zeros(n_heads, d_input, self.d_head)
        b1_head = torch.zeros(n_heads, self.d_head)
        w2_head = torch.zeros(n_heads, self.d_head, self.d_head_output)
        b2_head = torch.zeros(n_heads, self.d_head_output)
        _init(w1_head)
        _init(b1_head)
        _init(w2_head)
        _init(b2_head)
        self.genpool_w1_head = nn.Parameter(w1_head, requires_grad=True)
        self.genpool_b1_head = nn.Parameter(b1_head, requires_grad=True)
        self.genpool_w2_head = nn.Parameter(w2_head, requires_grad=True)
        self.genpool_b2_head = nn.Parameter(b2_head, requires_grad=True)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_temp = 1
        self.genpool_one = nn.Parameter(torch.ones(1), requires_grad=False)

    def extra_repr(self) -> str:
        strs = []
        for p in [
                self.genpool_w1_head, self.genpool_b1_head,
                self.genpool_w2_head, self.genpool_b2_head
        ]:
            strs.append(f"pool linear {p.shape}")
        return "\n".join(strs)

    def forward(self, features):
        # TODO: remove unused "lengths"
        #_ = lengths
        batch_size, seq_len, input_dim = features.shape
        # print(features.shape, self.genpool_w1_head.shape)
        b1 = torch.matmul(features.unsqueeze(1),
                          self.genpool_w1_head.unsqueeze(0))
        b1 += self.genpool_b1_head.unsqueeze(1).unsqueeze(0)
        b1 = self.activation(self.dropout1(b1))
        b1 = torch.matmul(b1, self.genpool_w2_head.unsqueeze(0))
        b1 += self.genpool_b2_head.unsqueeze(1).unsqueeze(0)
        b1 = self.dropout2(b1)
        # b1.masked_fill_((mask == 0).unsqueeze(1).unsqueeze(-1), -1e19)

        smweights = self.softmax(b1 / self.softmax_temp)
        smweights = self.dropout3(smweights)
        smweights = smweights.transpose(1, 2).reshape(-1, seq_len, input_dim)
        return (features * smweights).sum(dim=1) # pool features with attention weights


def _init_weight(w, init_gain=1):
    w.copy_(truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):
    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            _init_weight(val.data, init_std)
