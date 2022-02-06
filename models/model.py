from collections import OrderedDict
from easydict import EasyDict
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils_collection.general import truncated_normal_fill
from utils_collection.config_tools import TransformerConfig

from models.vit import ResEncoderBlockCTX,ResEncoderBlock, AttentionCTX, Block
from reformer_pytorch import LSHAttention
from axial_positional_embedding import AxialPositionalEmbedding
from .vit_tools import trunc_normal_

class ModalityFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x.permute(1, 0, 2)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]
        
class MultiExpertTransformer:
    def __init__(self, cfg: TransformerConfig, video_feature_name: List, modality_b, use_cuda: bool, use_multi_gpu: bool):
        self.use_cuda = use_cuda
        self.use_multi_gpu = use_multi_gpu
        self.model_list = []

        self.cfg: TransformerConfig = cfg
        # self.modality_list = video_feature_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        # self.device = 0 #dvc_cnt // 4

        if modality_b == "text":
            if len(video_feature_name) == 1:
                local_dim = cfg.dataset.modality_feat_dim[video_feature_name[0]]
            else:
                local_dim = cfg.dataset.feat_agg_dim
        else:
            if len(video_feature_name) == 2:
                local_dim = cfg.dataset.modality_feat_dim[video_feature_name[0]]
            else:
                local_dim = cfg.dataset.feat_agg_dim            

        self.modality_a_local = Transformer(cfg.model_cfgs['video_local'], local_dim)

        self.modality_a_local = self._to_device_fn(self.modality_a_local)
        # modality_b_name = cfg.dataset.modality_feat_name_b
        # print(modality_b_name)
        # print("=="*30)
        self.modality_b_local = Transformer(cfg.model_cfgs['text_local'], cfg.dataset.modality_feat_dim[modality_b])
        self.modality_b_local = self._to_device_fn(self.modality_b_local)

        # self.local_fusion = Transformer(cfg.model_cfgs['local_fusion'], cfg.model_cfgs['local_fusion'].output_dim)
        # self.local_fusion = self._to_device_fn(self.local_fusion)
        # self.modality_fusion = ModalityFusion(embed_dim=cfg.model_cfgs['video_local'].output_dim, 
        # num_heads=cfg.model_cfgs['video_local'].selfatn.num_heads,
        #  output_dim=cfg.model_cfgs['video_local'].output_dim)
        # self.modality_fusion = self._to_device_fn(self.modality_fusion)



        self.modality_a_global = Transformer(cfg.model_cfgs['video_global'],cfg.model_cfgs['video_global'].output_dim, use_context=True)
        self.modality_a_global = self._to_device_fn(self.modality_a_global)

        self.modality_b_global = Transformer(cfg.model_cfgs['text_global'],cfg.model_cfgs['text_global'].output_dim, use_context=True)
        self.modality_b_global = self._to_device_fn(self.modality_b_global)
        
        self.model_list = [
            self.modality_a_local, self.modality_a_global,
            self.modality_b_local, self.modality_b_global
        ]
    def unpack_data(self, i_mod, data_dict, use_cuda):
        def to_device(x):
            if use_cuda and isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            return x

        return [
            to_device(data_dict[i_mod][a])
            for a in ("vid_id", "vid_frames", "vid_frames_mask", "vid_frames_len",
                    "par_cap_vectors", "par_cap_mask", "par_cap_len", "clip_num",
                    "clip_frames", "clip_frames_len", "clip_frames_mask",
                    "sent_num", "sent_cap_vectors", "sent_cap_mask",
                    "sent_cap_len")
        ]

    def local_to_global(self, local_emb, local_num, net_name: str ):
        batch_size = len(local_num)
        max_local_len = torch.max(local_num)
        local_feat_dim = self.cfg.model_cfgs[net_name].output_dim
        local_emb_reshape = torch.zeros(
            (batch_size, max_local_len, local_feat_dim))
        local_emb_mask = torch.zeros((batch_size, max_local_len))
        local_emb_lens = torch.zeros((batch_size, ))
        if self.use_cuda:
            local_emb_reshape = local_emb_reshape.cuda(non_blocking=True)
            local_emb_mask = local_emb_mask.cuda(non_blocking=True)
            local_emb_lens = local_emb_lens.cuda(non_blocking=True)
        pointer = 0
        for batch, local_len in enumerate(local_num):
            local_emb_reshape[batch, :local_len, :] =\
                local_emb[pointer:pointer + local_len, :]
            local_emb_mask[batch, :local_len] = 1
            local_emb_lens[batch] = local_len
            pointer += local_len
        return local_emb_reshape


    def encode(self, data_dict):
        # (vid_id, vid_frames, vid_frames_mask, vid_frames_len,
        #     par_cap_vectors, par_cap_mask, par_cap_len, clip_num,
        #     modality_input_a, clip_frames_len, clip_frames_mask, sentence_num,
        #     modality_input_b, sentence_cap_mask,
        #     sentence_cap_len) = self.unpack_data("action", data_dict, self.use_cuda)
            
                    # input each modality into modality expert
        
            # print(i_modality, "feature dim: ", self.cfg.dataset.vid_feat_dim[i_modality])
        modality_input_a = data_dict["feat_modality_a"].cuda(self.device)
        modality_input_b = data_dict["feat_modality_b"].cuda(self.device)

        modality_input_global_a = data_dict["feat_modality_global_a"].cuda(self.device)
        modality_input_global_b = data_dict["feat_modality_global_b"].cuda(self.device)

        clip_num = data_dict["modality_num_a"].cuda(self.device)
        sentence_num = data_dict["modality_num_b"].cuda(self.device)


        # modality_input_a = modality_input_a.cuda(self.device)

        #====== Modality A ========
        emb_local_a = self.modality_a_local(modality_input_a)
        emb_local_ctx_a = self.modality_a_local(modality_input_global_a)
        global_input_a = self.local_to_global(emb_local_a, clip_num, net_name='video_local')
        global_emb_a= self.modality_a_global(global_input_a, emb_local_ctx_a, use_context=True)

        # print("text"*20)
        #====== Modality B ========
        emb_local_b = self.modality_b_local(modality_input_b)
        emb_local_ctx_b = self.modality_b_local(modality_input_global_b)
        global_input_b = self.local_to_global(emb_local_b, sentence_num, net_name='text_local')
        # compute paragraph embedding
        global_emb_b = self.modality_b_global(global_input_b, emb_local_ctx_b, use_context=True)


        # emb_local_a = emb_local_a.cuda(0)
            # print(clip_expert.shape, "====> ", i_modality)
        # clip_emb_b = []
        # for i_modality in self.modality_list:
        #     clip_emb_b.append(clip_emb_experts)
            # print(clip_emb.shape, clip_emb_experts[i_modality])
        # import pdb; pdb.set_trace()
        # clip_emb = clip_emb_experts #torch.stack(clip_emb_b, dim=1)
        # print("clip emb shape: ", clip_emb.shape)
        # clip_emb = clip_emb/len(self.modality_list)
        # clip_emb_fused = self.modality_fusion(clip_emb.cuda())
        # clip_emb_fused = clip_emb_fused.cuda()
        # print("clip emb fused shape: ", clip_emb_fused.shape)
        # print(video_global_input.shape, "Video Global")


        # print(clip_emb.shape,video_global_input.shape, sentence_emb.shape )

               

            
                #collect features and then input to fusion transformer
                #input the text POS taggs weights to fusion transformer

        # input the pooled features to global transformer. ! this can be optional.
        out_a = {"global_emb_a": global_emb_a, "local_emb_a":emb_local_a, "ctx_a":emb_local_ctx_a}
        out_b = {"global_emb_b": global_emb_b, "local_emb_b":emb_local_b, "ctx_b":emb_local_ctx_b}
        return out_a, out_b
        

    def encode_video(self, vid_frames, vid_frames_mask, vid_frames_len,
                     clip_num, clip_frames, clip_frames_len, clip_frames_mask):
        # compute video context
        # print(vid_frames.shape, vid_frames_mask.shape, vid_frames_len.shape)
        # print(vid_frames.shape, "input shape")

        vid_context, sub_ctx_vid = self.video_local(vid_frames, vid_frames_mask,
                                            vid_frames_len, None)
        # print(vid_context.shape, "video context shape")                                    
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
        # print(clip_frames.shape, "input to clip net")
        clip_emb, sub_clip = self.video_local(clip_frames, clip_frames_mask,
                                         clip_frames_len, None)
        # print(clip_emb.shape, "clip_emb shape")   
        # print("00"*20)                                 
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
        vid_emb, sub_vid = self.video_global(clip_emb_reshape, clip_emb_mask,
                                           clip_num, vid_context_hidden)
        # print(vid_emb.shape, sub_vid[0].shape, "--0-"*10)
        #TODO: convert the return to an object class or maybe dictionary
        return {"video_emb": vid_emb, "clip_emb":clip_emb, "video_ctx":vid_context,
                "sub_ctx_video":sub_ctx_vid, "sub_clip":sub_clip}

    def encode_paragraph(self, paragraph_caption_vectors, paragraph_caption_mask, paragraph_caption_len,
                         sentence_num, sentence_caption_vectors, sentence_caption_mask,
                         sentence_caption_len):
        # compute paragraph context
        paragraph_context, sub_ctx_paragraph = self.text_local(paragraph_caption_vectors, paragraph_caption_mask,
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
        sentence_emb, sub_sentence = self.text_local(sentence_caption_vectors, sentence_caption_mask,
                                        sentence_caption_len, None)
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
        paragraph_emb, sub_paragraph = self.text_global(sentence_emb_reshape, sentence_emb_mask,
                                          sentence_num, paragraph_gru_hidden)
        #TODO: convert the return to a object class or dictionary
        return {"paragraph_emb": paragraph_emb, "sentence_emb":sentence_emb, "paragraph_ctx":paragraph_context,
                "sub_ctx_paragraph":sub_ctx_paragraph, "sub_sentence":sub_sentence}
        # return (paragraph_emb, sentence_emb, paragraph_context, sentence_emb_reshape,
        #         sentence_emb_mask, sentence_emb_lens)

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
                    # print(key, val)
                    # assert not key.startswith("module.")
                    new_key = key #"module." + key
                    newer_state_dict[new_key] = val
                model.load_state_dict(newer_state_dict)
            else:
                model.load_state_dict(state_dict)
            i += 1 # we do this intentionally
            
    def get_model_state(self):
        model_states = []
        for m in self.model_list:
                state_dict = m.state_dict()
                # if self.use_multi_gpu:
                if False:
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
    def __init__(self, cfg: EasyDict, feature_dim: int, use_context=False):
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

        # print(cfg.use_input_fc, input_dim)
        self.tf = TransformerEncoder(cfg.selfatn.num_layers, input_dim, cfg.selfatn.num_heads,
                                     input_dim, cfg.selfatn.dropout)

        self.use_context = cfg.use_context
        if use_context:
            self.ctx_project = nn.Sequential(
                nn.Linear(cfg.selfatn.pointwise_ff_dim, input_dim), nn.GELU())
            self.tf_context = TransformerEncoder(cfg.selfatn.num_layers, input_dim, cfg.selfatn.num_heads,
                                     input_dim, cfg.selfatn.dropout)

        self.pooler = parse_pooler(input_dim, cfg)

        init_network(self, init_std=0.01)

    def forward(self, features, hidden_state=None, use_context=False):
        features = self.input_norm(features)
        if self.input_fc is not None:
            features = self.input_fc(features)
        features = self.embedding(features)
        features = self.tf(features, features, features)
        add_after_pool = None
        if use_context:
            hidden_state = hidden_state.unsqueeze(1)
            hidden_state = self.ctx_project(hidden_state)
            ctx = self.tf_context(hidden_state, features, features)
            add_after_pool = ctx.squeeze(1)
        pooled = self.pooler(features)
        if add_after_pool is not None:
            pooled = torch.cat([pooled, add_after_pool], dim=-1)
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

    def forward(self, query, key, value):
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, heads_count, fc_dim, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, model_dim, dropout_prob), model_dim)
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(fc_dim, model_dim, dropout_prob), model_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        sources = self.self_attention_layer(query, key, value)
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
        # print(query.shape, "====>")
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
        return torch.mean(features, dim=1)

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
        # print("b1 shape: ", b1.shape)
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

