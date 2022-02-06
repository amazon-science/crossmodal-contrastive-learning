import csv
from easydict import EasyDict
import os
from pathlib import Path
from timeit import default_timer as timer
from collections import defaultdict
import torch
import torch.nn.parallel
from torch.nn import functional as F
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from models.model import MultiExpertTransformer
from trainer.loss import ContrastiveLoss, ContrastiveLoss_coot, MaxMarginRankingLoss, NTXentLoss,CLIP, CLIP_Cluster, CrossCLR, CrossCLR_noq
from trainer.loss import DCL, MILNCELoss
from trainer.optimizer import get_optimizer, ReduceLROnPlateauWarmup
from utils_collection import general, metrics, yaml_config, compute_retrieval, config_tools as congt
from utils_collection.metrics import DefaultMetricsConst as Metrics
from utils_collection.config_tools import MetersConst 
from utils_collection.experiments import ExperimentFilesHandler
from trainer import trainer_base
import pickle as pk

def unpack_data(data_dict, use_cuda):
    def to_device(x):
        if use_cuda and isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        return x

    return [
        to_device(data_dict[a])
        for a in ("vid_id", "vid_frames", "vid_frames_mask", "vid_frames_len",
                  "par_cap_vectors", "par_cap_mask", "par_cap_len", "clip_num",
                  "clip_frames", "clip_frames_len", "clip_frames_mask",
                  "sent_num", "sent_cap_vectors", "sent_cap_mask",
                  "sent_cap_len")
    ]


class TrainerVideoText(trainer_base.BaseTrainer):
    def __init__(self, cfg,
                 video_feature_name: str,
                 second_modality_name: str,
                 exp_group: str,
                 exp_name: str, 
                 run_name: str, 
                 train_loader_length: int, 
                 log_dir: str = "experiments",
                 log_level = None,
                 logger=None,
                 reset: bool = False, 
                 load_ckpt: str = "",
                 save_emb_path: str = "",
                 is_train=True):
        super().__init__(
            cfg, exp_group, exp_name, run_name, train_loader_length, 'retrieval',
            log_dir=log_dir, log_level=log_level, logger=logger, reset=reset,
            is_test= not is_train)

        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.use_multi_gpu = self.cfg.use_multi_gpu
        self.use_cuda = self.cfg.use_cuda

        self.timer_start_train = 0
        self.det_best_field_current = 0
        self.det_best_field_best = 0
        self.best_epoch = 0
        self.modality_list = video_feature_name
        self.modality_b = second_modality_name
        self.save_emb_path = save_emb_path
        self.path_ckpt = load_ckpt

        ###================================================
        # ---------- additional metrics ----------
        # loss proportions
        self.metrics.add_meter(MetersConst.VAL_LOSS_CONTRASTIVE, use_avg=False)
        self.metrics.add_meter(MetersConst.TEST_LOSS_CONTRASTIVE, use_avg=False)
        self.metrics.add_meter(MetersConst.TRAIN_LOSS_CONTRASTIVE, per_step=True, use_avg=False)

        # retrieval validation metrics must be constructed as product of two lists
        for modality in MetersConst.RET_MODALITIES:
            # modality: retrieval from where to where
            for metric in MetersConst.RET_METRICS:
                # metric: retrieval@1, mean, ...
                if metric == "r1":
                    # log r1 metric to the overview class
                    metric_class = "val_base"
                else:
                    # log all other metrics to the detail class
                    metric_class = "val_ret"
                self.metrics.add_meter(f"{metric_class}/{modality}-{metric}", use_avg=False)

        # retrieval validation metrics must be constructed as product of two lists
        for modality in MetersConst.RET_MODALITIES:
            # modality: retrieval from where to where
            for metric in MetersConst.RET_METRICS:
                # metric: retrieval@1, mean, ...
                if metric == "r1":
                    # log r1 metric to the overview class
                    metric_class = "test_base"
                else:
                    # log all other metrics to the detail class
                    metric_class = "test_ret"
                self.metrics.add_meter(f"{metric_class}/{modality}-{metric}", use_avg=False)


        # model
        self.model = MultiExpertTransformer(cfg, video_feature_name, self.modality_b, self.cfg.use_cuda, self.use_multi_gpu)
        params, _param_names, _params_flat = self.model.get_all_params()

        model_params_num = sum(p.numel() for p in _params_flat if p.requires_grad)
        if model_params_num // 10 ** 6 > 0:
            self.logger.info('===> Model total parameter: {} M \n'.format(model_params_num // 10 ** 6))
        else:
            self.logger.info('===> Model total parameter: {} K \n'.format(model_params_num // 10 ** 3))


        # contrastive loss
        # self.loss_f_contr = NTXentLoss(0.04, 1) #ContrastiveLoss(use_cuda)
        self.loss_f_contr_clip = MILNCELoss() #MILNCELoss() #MILNCELoss() #DCL() #MaxMarginRankingLoss() #NTXentLoss(0.04, 1) #CLIP(0.05) #NTXentLoss(0.04, 1)
        self.loss_f_contr_vid =  MILNCELoss() #MILNCELoss() #MILNCELoss() #DCL() #MaxMarginRankingLoss() #NTXentLoss(0.04, 1) #CLIP(0.05) #NTXentLoss(0.04, 1)

        # self.loss_f_contr = CLIP(0.05)
        # self.loss_f_contr_clip = CLIP(0.06) #CLIP(0.07) #NTXentLoss(0.04, 1) #MaxMarginRankingLoss() #CLIP(0.05)
        if len(self.modality_list) > 1:
            q_agg_dim = self.cfg.dataset.feat_agg_dim
        else:
            q_agg_dim = self.cfg.dataset.modality_feat_dim[self.modality_list[0]]

        # self.loss_f_contr_clip = CrossCLR(self.cfg.train.cross_clr_config.temperature, self.cfg.train.cross_clr_config.temperature_weights,
        # self.cfg.train.cross_clr_config.negative_weight, self.cfg.train.cross_clr_config.score_thrshold, 
        # self.cfg.model_cfgs["video_local"].output_dim, self.cfg.train.cross_clr_config.queue_size,
        # q_agg_dim, self.cfg.dataset.modality_feat_dim[self.modality_b], self.logger)

        # self.loss_f_contr_clip = CrossCLR_noq(self.cfg.train.cross_clr_config.temperature, self.cfg.train.cross_clr_config.temperature_weights,
        # self.cfg.train.cross_clr_config.negative_weight, self.cfg.train.cross_clr_config.score_thrshold, self.logger)
        
        
        # self.loss_f_contr_clip = CrossCLR_noq(self.cfg.train.cross_clr_config.temperature, self.cfg.train.cross_clr_config.temperature_weights,
        # self.cfg.train.cross_clr_config.negative_weight, self.cfg.train.cross_clr_config.score_thrshold, self.logger)

        # self.loss_f_contr_vid = CrossCLR_noq(self.cfg.train.cross_clr_config.temperature, self.cfg.train.cross_clr_config.temperature_weights,
        # self.cfg.train.cross_clr_config.negative_weight, self.cfg.train.cross_clr_config.score_thrshold, self.logger)
        
        # self.loss_f_contr_vid = CrossCLR(self.cfg.train.cross_clr_config.temperature, self.cfg.train.cross_clr_config.temperature_weights,
        # self.cfg.train.cross_clr_config.negative_weight, self.cfg.train.cross_clr_config.score_thrshold, 
        # 1024, self.cfg.train.cross_clr_config.queue_size,
        # 2048, 1536, self.logger)



        # self.loss_f_contr_vid = MaxMarginRankingLoss() #MaxMarginRankingLoss() #ContrastiveLoss(self.cfg.use_cuda)  #CLIP(0.07) #MaxMarginRankingLoss() #ContrastiveLoss(self.cfg.use_cuda) #CLIP(0.05)
        # self.loss_f_contr = CLIP_Cluster(0.05, self.logger)

        # self.loss_f_contr_cluster=  MaxMarginRankingLoss() #ContrastiveLoss_coot(use_cuda) #
        #print("Start using SimCLR and MaxMargin for clip clustering loss")
        # self.loss_f_contr = ContrastiveLoss_coot(use_cuda)
        # self.loss_f_contr = ContrastiveLoss(self.cfg.use_cuda) #with hard negative sampling

        # self.loss_f_contr = ContrastiveLoss_dw(use_cuda)
        # self.loss_f_contr = MaxMarginRankingLoss()
        #print("Start using MaxMargin loss")


        # optimizer
        self.optimizer = get_optimizer(cfg.optimizer, params)

        # scheduler
        # import pdb
        # pdb.set_trace()
        self.lr_scheduler = ReduceLROnPlateauWarmup(
            self.optimizer,
            cfg.lr_scheduler.warmup_epochs,
            mode="max",
            patience=cfg.lr_scheduler.rop_patience,
            cooldown=cfg.lr_scheduler.rop_cooldown)

        if load_ckpt != "":
            self.logger.info(f"Load checkpoint {load_ckpt}")
            self.model.load_checkpoint(load_ckpt)

    def compare_metrics(self, comparison, best):
        if best is None:
            return True
        threshold = 1e-4
        rel_epsilon = threshold + 1.
        return comparison > best * rel_epsilon

    def compute_align_loss_clip(self, v_embedding, p_embedding, input_a_org=None, input_b_org=None):
        return self.loss_f_contr_clip(v_embedding, p_embedding) #, input_a_org, input_b_org)

    def compute_align_loss_vid(self, v_embedding, p_embedding, input_a=None, input_b=None):
        return self.loss_f_contr_vid(v_embedding, p_embedding) #, input_a, input_b)

    def compute_align_loss_expert(self, expert_embeddings, text_embedding):
        loss_expert = 0
        for i_modality in self.modality_list:
            loss_expert += self.loss_f_contr_clip(expert_embeddings[i_modality], text_embedding)
        return loss_expert / len(self.modality_list)

    def compute_cluster_loss(self, v_embedding, p_embedding):
        return (self.loss_f_contr_cluster(v_embedding, v_embedding) +
                self.loss_f_contr_cluster(p_embedding, p_embedding)) / 2

    # def compute_total_constrastive_loss(self, vid_emb, par_emb, clip_emb,
    #                                     sent_emb, vid_context, par_context):
    #     vid_context_norm = F.normalize(vid_context)
    #     clip_emb_norm = F.normalize(clip_emb)
    #     vid_emb_norm = F.normalize(vid_emb)
    #     par_context_norm = F.normalize(par_context)
    #     sent_emb_norm = F.normalize(sent_emb)
    #     par_emb_norm = F.normalize(par_emb)

    #     # vid_context_norm = vid_context
    #     # clip_emb_norm = clip_emb
    #     # vid_emb_norm = vid_emb
    #     # par_context_norm = par_context
    #     # sent_emb_norm = sent_emb
    #     # par_emb_norm = par_emb

    #     loss = self.cfg.train.contrastive_loss_config.weight_high * self.compute_align_loss_vid(vid_emb_norm, par_emb_norm)
    #     loss += self.cfg.train.contrastive_loss_config.weight_low * self.compute_align_loss_clip(clip_emb_norm, sent_emb_norm)
    #     loss += self.cfg.train.contrastive_loss_config.weight_context * self.compute_align_loss_vid(vid_context_norm, par_context_norm)
    #     loss += self.cfg.train.contrastive_loss_config.weight_high_cluster * self.compute_cluster_loss(vid_emb_norm, par_emb_norm)
    #     loss += self.cfg.train.contrastive_loss_config.weight_low_cluster * self.compute_cluster_loss(clip_emb_norm, sent_emb_norm)
    #     return loss

    def compute_align_loss_sub(self, v_embedding, p_embedding):
        return sum([self.loss_f_contr_clip(x_video, x_text) for x_video, x_text in zip(v_embedding, p_embedding)])/len(v_embedding)

    def compute_align_loss_selfsub(self, sub_emb, main_emb):
        # print(sub_emb[0].shape, main_emb.shape, "===>")
        return sum([self.loss_f_contr_clip(x_sub, main_emb) for x_sub in sub_emb])/len(sub_emb)

    def compute_total_constrastive_loss(self, emb_dict_a, emb_dict_b, data_dict):

        # out_a = {"global_emb_a": global_emb_a, "emb_local_a":emb_local_a}
        # out_b = {"global_emb_b": global_emb_b, "emb_local_b":emb_local_b}
        # vid_context_norm = F.normalize(video_emb_dict["video_ctx"])
        emb_local_norm_a = F.normalize(emb_dict_a["local_emb_a"])
        emb_global_norm_a = F.normalize(emb_dict_a["global_emb_a"])

        emb_local_norm_b = F.normalize(emb_dict_b["local_emb_b"])
        emb_global_norm_b = F.normalize(emb_dict_b["global_emb_b"])
        
        ctx_norm_a = F.normalize(emb_dict_a["ctx_a"])
        ctx_norm_b = F.normalize(emb_dict_b["ctx_b"])

        clip_expert = {}
        modality_a_list = [] 
        modality_b_list = []

        dim_feats = 0
        # for i_modality in self.modality_list:
        #     dim_feats += data_dict[i_modality]["clip_frames"].shape[2]

        # modality_a = torch.empty(size=(data_dict[i_modality]["clip_frames"].shape[0], dim_feats))

        # idx_mod = 0 
        # for i_modality in self.modality_list:
        #     # clip_expert[i_modality] = F.normalize(clip_emb_experts)
        #     # print(data_dict[i_modality]["clip_frames"].mean(dim=1).shape)
        #     # print(data_dict[i_modality]["sent_cap_vectors"].mean(dim=1).shape)
        #     dim_modality = data_dict[i_modality]["clip_frames"].mean(dim=1).shape[1]
        #     modality_a[:,idx_mod:idx_mod+ dim_modality] = F.normalize(data_dict[i_modality]["clip_frames"].mean(dim=1))
        #     idx_mod += dim_modality


        # modality_a = F.normalize(data_dict["feat_modality_a"].mean(dim=1))
        # modality_b = F.normalize(data_dict["feat_modality_b"].mean(dim=1))

        modality_a = data_dict["feat_modality_a"].mean(dim=1)
        modality_b = data_dict["feat_modality_b"].mean(dim=1)

        input_global_a = data_dict["feat_modality_global_a"].mean(dim=1)
        input_global_b = data_dict["feat_modality_global_b"].mean(dim=1)

        # modality_a = emb_local_norm_a
        # modality_b = emb_local_norm_b

        # print(modality_a.shape, modality_b.shape)

        # subctx_video_norm = [F.normalize(x) for x in video_emb_dict["sub_ctx_video"]]
        # subclip_norm = [F.normalize(x) for x in video_emb_dict["sub_clip"]]
        # print(subctx_video_norm[0].shape, "subctx_video_norm")
        # print(subclip_norm[0].shape, "subclip_norm")

        # par_context_norm = F.normalize(text_emb_dict["paragraph_ctx"])
        # emb_local_norm_b = F.normalize(emb_dict_b["local_emb_b"])
        # emb_global_norm_b = F.normalize(emb_dict_b["global_emb_b"])
        # subctx_text_norm = [F.normalize(x) for x in text_emb_dict["sub_ctx_paragraph"]]
        # subsentence_norm = [F.normalize(x) for x in text_emb_dict["sub_sentence"]]
        # print(subctx_text_norm[0].shape, "subctx_text_norm")
        # print(subsentence_norm[0].shape, "subsentence_norm")

        # vid_context_norm = vid_context
        # clip_emb_norm = clip_emb
        # vid_emb_norm = vid_emb
        # par_context_norm = par_context
        # sent_emb_norm = sent_emb
        # par_emb_norm = par_emb

        loss = self.cfg.train.contrastive_loss_config.weight_high * self.compute_align_loss_vid(emb_global_norm_a, emb_global_norm_b, input_global_a, input_global_b)
        loss += self.cfg.train.contrastive_loss_config.weight_low * self.compute_align_loss_clip(emb_local_norm_a, emb_local_norm_b, modality_a, modality_b)
        # loss += self.cfg.train.contrastive_loss_config.weight_context * self.compute_align_loss_vid(ctx_norm_a, ctx_norm_b, input_global_a, input_global_b)

        # loss += self.cfg.train.contrastive_loss_config.weight_high_cluster * self.compute_cluster_loss(vid_emb_norm, par_emb_norm)
        # loss += self.cfg.train.contrastive_loss_config.weight_low_cluster * self.compute_cluster_loss(clip_emb_norm, sent_emb_norm)
        
        # loss += self.cfg.train.contrastive_loss_config.weight_low * self.compute_align_loss_expert(clip_expert, sent_emb_norm)
        # loss += self.cfg.train.contrastive_loss_config.weight_low * self.compute_align_loss_clip(clip_scene, sent_emb_norm)

        # loss += self.compute_align_loss_sub(subctx_video_norm, subctx_text_norm)
        # loss += 0.3 * self.compute_align_loss_sub(subclip_norm, subsentence_norm)

        
        # loss += self.compute_align_loss_selfsub(subclip_norm, sent_emb_norm)
        # loss += self.compute_align_loss_selfsub(subsentence_norm, clip_emb_norm)
        # loss += self.compute_align_loss_selfsub(subctx_video_norm, sent_emb_norm)
        # loss += self.compute_align_loss_selfsub(subctx_text_norm, clip_emb_norm)
        return loss

    # def close(self):
        
    #     if self.metrics_fh is not None:
    #         self.metrics_fh.close()
    #     general.close_logger(self.logger)

    def unpack_aggregate_data(self, modality_list, second_modality, data_dict, use_cuda):
        def to_device(x):
            if use_cuda and isinstance(x, torch.Tensor):
                return x.cuda(non_blocking=True)
            return x
        self.cfg.dataset.feat_agg_dim

        def aggregate(modality_list, second_modality, data_name):
            # assert self.cfg.dataset.feat_agg_dim == 2048, "Currently we only support feature agg dim 512"
            
            if second_modality == "text":
                first_modality = modality_list
            else:
                first_modality = modality_list[:-1]

            mods = []
            for i_mod in first_modality:

                if data_dict[i_mod][data_name].shape[-1] > self.cfg.dataset.feat_agg_dim:
                    n_div = data_dict[i_mod][data_name].shape[-1] // self.cfg.dataset.feat_agg_dim
                    tmp = data_dict[i_mod][data_name][:, :, :n_div*self.cfg.dataset.feat_agg_dim].reshape(data_dict[i_mod][data_name].shape[0], -1, n_div, self.cfg.dataset.feat_agg_dim)\
                    .permute(0,2,1,3).reshape(data_dict[i_mod][data_name].shape[0],-1, self.cfg.dataset.feat_agg_dim)
                    # print(tmp.shape, data_dict[i_mod][data_name].shape[-1])
                elif data_dict[i_mod][data_name].shape[-1] < self.cfg.dataset.feat_agg_dim:
                    n_div = self.cfg.dataset.feat_agg_dim // data_dict[i_mod][data_name].shape[-1]
                    if n_div * data_dict[i_mod][data_name].shape[-1] < self.cfg.dataset.feat_agg_dim:
                        n_div += 1
                    round_dim = self.cfg.dataset.feat_agg_dim // n_div
                    tmp = data_dict[i_mod][data_name][:, :, :round_dim].repeat(1,1,n_div)
                    # print(tmp.shape, n_div, round_dim, data_dict[i_mod][data_name].shape[-1])
                else:
                    tmp = data_dict[i_mod][data_name]
                    # print(tmp.shape, data_dict[i_mod][data_name].shape[-1])

                mods.append(tmp)

                    # if i_mod == "scene":
                    #     # tmp = data_dict[i_mod][data_name][:, :, :2048].reshape(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)
                    #     # n_div = data_dict[i_mod][data_name][:, :, :2048].shape[-1] // self.cfg.dataset.feat_agg_dim
                    #     tmp = data_dict[i_mod][data_name][:, :, :2048] #.reshape(data_dict[i_mod][data_name].shape[0], -1, n_div, self.cfg.dataset.feat_agg_dim)\
                    #     # .permute(0,2,1,3).reshape(data_dict[i_mod][data_name].shape[0],-1, self.cfg.dataset.feat_agg_dim)
                    # elif i_mod == "flow":
                    #     tmp = data_dict[i_mod][data_name][:, :, :512].repeat(1,1,4) #.view(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)

                    # elif i_mod == "object":
                    #     # import pdb; pdb.set_trace()
                    #     tmp = data_dict[i_mod][data_name].repeat(1,1,8)
                    # else:
                    #     # tmp = data_dict[i_mod][data_name].view(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)
                    #     # n_div = data_dict[i_mod][data_name].shape[-1] // self.cfg.dataset.feat_agg_dim
                    #     tmp = data_dict[i_mod][data_name] #.reshape(data_dict[i_mod][data_name].shape[0], -1, n_div, self.cfg.dataset.feat_agg_dim)\
                    #     # .permute(0,2,1,3).reshape(data_dict[i_mod][data_name].shape[0],-1, self.cfg.dataset.feat_agg_dim)
                    
                    # mods.append(tmp)
            # else:
            #     mods = []
            #     for i_mod in modality_list[:-1]:
            #         if i_mod == "scene":
            #             # tmp = data_dict[i_mod][data_name][:, :, :2048].reshape(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)
            #             # n_div = data_dict[i_mod][data_name][:, :, :2048].shape[-1] // self.cfg.dataset.feat_agg_dim
            #             tmp = data_dict[i_mod][data_name][:, :, :2048] #.reshape(data_dict[i_mod][data_name].shape[0], -1, n_div, self.cfg.dataset.feat_agg_dim)\
            #             # .permute(0,2,1,3).reshape(data_dict[i_mod][data_name].shape[0],-1, self.cfg.dataset.feat_agg_dim)

            #         elif i_mod == "flow":
            #             tmp = data_dict[i_mod][data_name][:, :, :512].repeat(1,1,2) #.view(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)

            #         elif i_mod == "object":
            #             # import pdb; pdb.set_trace()
            #             tmp = data_dict[i_mod][data_name].repeat(1,1,4)
            #         else:
            #             # tmp = data_dict[i_mod][data_name].view(data_dict[i_mod][data_name].shape[0], -1, self.cfg.dataset.feat_agg_dim)
            #             # n_div = data_dict[i_mod][data_name].shape[-1] // self.cfg.dataset.feat_agg_dim
            #             tmp = data_dict[i_mod][data_name] #.reshape(data_dict[i_mod][data_name].shape[0], -1, n_div, self.cfg.dataset.feat_agg_dim)\
            #             # .permute(0,2,1,3).reshape(data_dict[i_mod][data_name].shape[0],-1, self.cfg.dataset.feat_agg_dim)
                    
            #         mods.append(tmp)
            modality_a = torch.cat(mods, dim=1)
            return modality_a

        # else:
        data_p = {}

        if second_modality == "text":
            data_p["modality_num_a"] = to_device(data_dict[modality_list[0]]["clip_num"])
            data_p["feat_modality_b"] = (to_device(data_dict[modality_list[0]]["sent_cap_vectors"]))
            data_p["feat_modality_global_b"] = (to_device(data_dict[modality_list[0]]["par_cap_vectors"]))
            data_p["modality_num_b"] = to_device(data_dict[modality_list[0]]["sent_num"])
            if len(modality_list) > 1:
                data_p["feat_modality_a"] = to_device(aggregate(modality_list, second_modality, "clip_frames"))
                data_p["feat_modality_global_a"] = to_device(aggregate(modality_list, second_modality, "vid_frames"))
            else:
                data_p["feat_modality_a"] = to_device(data_dict[modality_list[0]]["clip_frames"])
                data_p["feat_modality_global_a"] = to_device(data_dict[modality_list[0]]["vid_frames"])

        else:
            data_p["modality_num_a"] = to_device(data_dict[modality_list[0]]["clip_num"])
            data_p["feat_modality_b"] = (to_device(data_dict[modality_list[-1]]["clip_frames"]))
            data_p["feat_modality_global_b"] = (to_device(data_dict[modality_list[-1]]["vid_frames"]))
            data_p["modality_num_b"] = to_device(data_dict[modality_list[-1]]["clip_num"])
            # data_dic.append(aggregate(modality_list, second_modality, i_data_name))
            if len(modality_list) > 2:
                data_p["feat_modality_a"] = to_device(aggregate(modality_list, second_modality, "clip_frames"))
                data_p["feat_modality_global_a"] = to_device(aggregate(modality_list, second_modality, "vid_frames"))
            else:
                data_p["feat_modality_a"] = to_device(data_dict[modality_list[0]]["clip_frames"])
                data_p["feat_modality_global_a"] = to_device(data_dict[modality_list[0]]["vid_frames"])
        return data_p


        # return [
        #     to_device(data_dict[i_mod][a])
        #     for a in ("vid_id", "vid_frames", "vid_frames_mask", "vid_frames_len",
        #             "par_cap_vectors", "par_cap_mask", "par_cap_len", "clip_num",
        #             "clip_frames", "clip_frames_len", "clip_frames_mask",
        #             "sent_num", "sent_cap_vectors", "sent_cap_mask",
        #             "sent_cap_len")
        # ]d

    def train_loop(self, train_loader, val_loader, test_loader=None):

        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch
        
        max_step = len(train_loader)
        self.timer_start_train = timer()
        epoch = 0
        for epoch in range(0, self.cfg.train.num_epochs):
            if self.check_early_stop():
                break
            self.hook_pre_train_epoch()  # pre-epoch hook: set models to train, time book-keeping

            self.model.train()
            

            # train one epoch
            self.logger.info(
                "---------- Training epoch {} ----------".format(epoch))
            for step, data_dict in enumerate(train_loader):
                if step == 0:
                   self.logger.info(" Modalities: {}".format(self.modality_list))
                   self.logger.info(" Second modality: {}".format(self.modality_b))
                   self.logger.info(f"First step data ids: {data_dict[self.modality_list[0]]['vid_id'][:4]}...")

                self.hook_pre_step_timer()  # hook for step timing
                self.optimizer.zero_grad()


                # Check modality list in dataloader: modalities=data_dict[-1]
                data_p = self.unpack_aggregate_data(self.modality_list, self.modality_b, data_dict, self.use_cuda)
                # print(self.modality_b, data_p["feat_modality_b"].shape, data_p["feat_modality_a"].shape)
                video_emb_dict, text_emb_dict = self.model.encode(data_dict=data_p)

                # text_emb_dict = self.model.encode_paragraph(data_dict[0])

                loss = self.compute_total_constrastive_loss(video_emb_dict, text_emb_dict, data_dict=data_p)


                self.hook_post_forward_step_timer()  # hook for step timing
                # backward pass

                # ---------- backward pass ----------
                # print("===> fp16_train", self.cfg.fp16_train)
                if self.cfg.fp16_train:
                    # with fp16 amp
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    # with regular float32
                    loss.backward()
                    self.optimizer.step()

                cc_loss = 0
                additional_log = f"L Contr: {loss:.5f}, L CC: {cc_loss:.5f}"
                self.hook_post_backward_step_timer()  # hook for step timing

                # post-step hook: gradient clipping, profile gpu, update metrics, count step, step LR scheduler, log
                self.hook_post_step(step, loss, self.optimizer.param_groups[0]['lr'], additional_log=additional_log)




            # ---------- validation ----------
            is_best = False
            do_val = self.check_is_val_epoch()
            if do_val:
                # validate one epoch
                self.logger.info(
                "---------- Validating epoch {} ----------".format(epoch))
                vid_metrics, clip_metrics, is_best = self.validate(val_loader)
                v2p_res, p2v_res, vid_best_at_1 = vid_metrics
                c2s_res, s2c_res, clip_best_at_1 = None, None, None
                if clip_metrics is not None:
                    c2s_res, s2c_res, clip_best_at_1 = clip_metrics

                # find field which determines is_best
                if self.cfg.val.det_best_field == "val_score_at_1":
                    self.det_best_field_current = vid_best_at_1
                elif self.cfg.val.det_best_field == "val_clip_score_at_1":
                    self.det_best_field_current = clip_best_at_1
                else:
                    raise NotImplementedError

            if is_best:
                self.det_best_field_best = self.det_best_field_current
                self.best_epoch = epoch

            self.hook_post_train_and_val_epoch(do_val, is_best)

            # step lr scheduler
            self.lr_scheduler.step_rop(self.det_best_field_current, True)
            self.logger.info(
                f"ROP: model improved: {is_best}, "
                f"value {self.det_best_field_current:.3f},"
                f"new LR: {self.optimizer.param_groups[0]['lr']:5.3e}")

            # -------------- Test ----------------------
            # test one epoch
            do_test = self.check_is_test_epoch()
            if do_test and test_loader is not None:
                self.logger.info(
                "------ ---- ------ ------ --- ---- ---- Test epoch {} - --- --- -- ---------".format(epoch))
                self.test(test_loader)

            # check if model did not improve for too long
            term_after = 15
            if epoch - self.best_epoch > term_after:
                self.logger.info(
                    f"NO improvements for {term_after} epochs (current "
                    f"{epoch} best {self.best_epoch}) STOP training.")
                break

        time_total = timer() - self.timer_start_train

        # show end of training log message
        self.hook_post_train()


    def _normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def validate(self, val_loader, debug_max=-1):
        self.model.eval()

        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        loss_total: th.Tensor = 0.
        contr_loss_total: th.Tensor = 0.
        cc_loss_total: th.Tensor = 0.
        data_collector = {}

        max_step = len(val_loader)
        num_steps = 0


        # collect embeddings
        vid_emb_list = []
        par_emb_list = []
        clip_emb_list = []
        sent_emb_list = []

        clip_emb_list1 = []
        sent_emb_list1 = []

        for step, data_dict in enumerate(val_loader):
            if step >= debug_max > -1:
                break
            # (vid_id, vid_frames, vid_frames_mask, vid_frames_len,
            #  par_cap_vectors, par_cap_mask, par_cap_len, clip_num, clip_frames,
            #  clip_frames_len, clip_frames_mask, sent_num, sent_cap_vectors,
            #  sent_cap_mask,
            #  sent_cap_len) = unpack_data(data_dict, self.use_cuda)
            if step == 0:
                print(f"ids {data_dict[self.modality_list[0]]['vid_id'][:4]}...")
                
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing
            data_p = self.unpack_aggregate_data(self.modality_list, self.modality_b, data_dict, self.use_cuda)

            video_emb_dict, text_emb_dict = self.model.encode(data_dict=data_p)

            loss = self.compute_total_constrastive_loss(video_emb_dict, text_emb_dict, data_dict=data_p)

            loss_total += loss.item()

            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            # collect embeddings
            # print(video_emb_dict["video_emb"].shape)
            vid_emb_list.extend(video_emb_dict["global_emb_a"].detach().cpu())
            par_emb_list.extend(text_emb_dict["global_emb_b"].detach().cpu())
            
            # xs = 0
            # xv = 0 

            # for vx, sx in zip(video_emb_dict["sub_clip"], text_emb_dict["sub_sentence"]):
            #     xv += vx
            #     xs += sx        

            # emN1 = (video_emb_dict["clip_emb"] + xv / 4) / 4
            # emN2 = (text_emb_dict["sentence_emb"] + xs/4) / 4

            # clip_emb_list1.extend(emN1.detach().cpu())
            # sent_emb_list1.extend(emN2.detach().cpu())

            clip_emb_list.extend(video_emb_dict["local_emb_a"].detach().cpu())
            sent_emb_list.extend(text_emb_dict["local_emb_b"].detach().cpu())


            # clip_emb_list_sub.extend(video_emb_dict["clip_emb"].detach().cpu())  
            # sent_emb_list_sub.extend(text_emb_dict["sentence_emb"].detach().cpu())             

            # logging
            if step % 10 == 0:
                self.logger.info(
                    f"Val [{step}/{max_step}] Loss {loss.item():.4f}")
        vid_emb_list = torch.stack(vid_emb_list, 0)
        par_emb_list = torch.stack(par_emb_list, 0)

        clip_emb_list = torch.stack(clip_emb_list, 0)
        sent_emb_list = torch.stack(sent_emb_list, 0)

        # clip_emb_list1 = torch.stack(clip_emb_list1, 0)
        # sent_emb_list1 = torch.stack(sent_emb_list1, 0)
        # video text retrieval
        vid_emb_list = F.normalize(vid_emb_list).numpy()
        par_emb_list = F.normalize(par_emb_list).numpy()

        # print(len(clip_emb_list), clip_emb_list[0])
        clip_emb_list = F.normalize(clip_emb_list).numpy()
        sent_emb_list = F.normalize(sent_emb_list).numpy()

        # clip_emb_list1 = F.normalize(clip_emb_list1).numpy()
        # sent_emb_list1 = F.normalize(sent_emb_list1).numpy()

        # print("--0--"*20)
        # results = torch.cat(clip_emb_list_sub, dim=0)

        # calculate total loss and feed meters
        loss_total /= num_steps
        # print("=="*20, loss_total, loss)
        forward_time_total /= num_steps
        self.metrics.update_meter(MetersConst.VAL_LOSS_CONTRASTIVE, loss_total)

        # calculate video-paragraph retrieval and print output table
        self.logger.info(compute_retrieval.VALHEADER)

        v2p_res, p2v_res, sum_vp_at_1, str_vp = compute_retrieval.compute_scores(
            vid_emb_list, par_emb_list, "vid_emb", "par_emb", print_fn=self.logger.info)

        c2s_res, s2c_res, sum_cs_at_1, str_cs = compute_retrieval.compute_scores(
                clip_emb_list, sent_emb_list, "clip_emb", "sent_emb", print_fn=self.logger.info)

        # c2s_res1, s2c_res1, sum_cs_at_11, str_cs1 = compute_retrieval.compute_scores(
        #         clip_emb_list1, sent_emb_list1, "clip_emb", "sent_emb", print_fn=self.logger.info)



        c2s_sum_at_1 = c2s_res["r1"] + s2c_res["r1"]

        sum_at_1 = v2p_res["r1"] + p2v_res["r1"]


        # feed retrieval results to meters
        for modality, dict_ret in zip(MetersConst.RET_MODALITIES, [v2p_res, p2v_res, c2s_res, s2c_res]):
            if dict_ret is None:
                continue
            # iterate over result keys
            for metric in MetersConst.RET_METRICS:
                # feed averagemeters
                logger_class = "val_ret"
                if metric == "r1":
                    logger_class = "val_base"
                self.metrics.update_meter(f"{logger_class}/{modality}-{metric}", dict_ret[metric])

        # print some more details about the retrieval (time, number of datapoints)
        self.logger.info(
            f"Loss {loss_total:.5f} "
            f"Retrieval: {str_vp}{str_cs}total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s")

        # find field which determines whether this is a new best epoch
        if self.cfg.val.det_best_field == "val_score_at_1":
            val_score = sum_vp_at_1
        elif self.cfg.val.det_best_field == "val_loss":
            val_score = loss_total
        elif self.cfg.val.det_best_field == "val_clip_score_at_1":
            val_score = sum_cs_at_1
        else:
            raise NotImplementedError(f"best field {self.cfg.val.det_best_field} not known")

        # check for a new best epoch and update validation results
        is_best = self.check_is_new_best(val_score)
        self.hook_post_val_epoch(loss_total, is_best)

        # if self.is_test:
        #     # for test runs, save the validation results separately to a file
        #     self.metrics.feed_metrics(False, self.state.total_step, self.state.current_epoch)
        #     metrics_file = self.exp.path_base / f"val_ep_{self.state.current_epoch}.json"
        #     self.metrics.save_epoch_to_file(metrics_file)
        #     self.logger.info(f"Saved validation results to {metrics_file}")

        return (v2p_res, p2v_res, sum_at_1), (c2s_res, s2c_res, c2s_sum_at_1), is_best

    def test(self, test_loader, debug_max=-1):
        self.model.eval()

        self.hook_pre_val_epoch()  # pre val epoch hook: set models to val and start timers
        forward_time_total = 0
        loss_total: th.Tensor = 0.
        contr_loss_total: th.Tensor = 0.
        cc_loss_total: th.Tensor = 0.
        data_collector = {}

        max_step = len(test_loader)
        num_steps = 0


        # collect embeddings
        vid_emb_list = []
        par_emb_list = []
        clip_emb_list = []
        sent_emb_list = []

        clip_emb_list1 = []
        sent_emb_list1 = []
        save_sent_vec, save_sent_cap, save_clip_vec, save_key = [], [], [], []

        for step, data_dict in enumerate(test_loader):
            if step >= debug_max > -1:
                break
            if step == 0:
                print(f"ids {data_dict[self.modality_list[0]]['vid_id'][:4]}...")
                
            # ---------- forward pass ----------
            self.hook_pre_step_timer()  # hook for step timing
            data_p = self.unpack_aggregate_data(self.modality_list, self.modality_b, data_dict, self.use_cuda)
            # import pdb; pdb.set_trace()
            if self.save_emb_path != "":
                # collect meta information for saving
                save_clip_vec.extend(data_dict[self.modality_list[0]]["clip_frames"].cpu().numpy().tolist())
                save_sent_vec.extend(data_dict[self.modality_list[0]]["sent_cap_vectors"].cpu().numpy().tolist())
                save_sent_cap.extend(data_dict[self.modality_list[0]]["data_words"])
                save_key.extend(data_dict[self.modality_list[0]]["vid_id"])


            video_emb_dict, text_emb_dict = self.model.encode(data_dict=data_p)

            loss = self.compute_total_constrastive_loss(video_emb_dict, text_emb_dict, data_dict=data_p)

            loss_total += loss.item()

            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            # collect embeddings
            # print(video_emb_dict["video_emb"].shape)
            vid_emb_list.extend(video_emb_dict["global_emb_a"].detach().cpu())
            par_emb_list.extend(text_emb_dict["global_emb_b"].detach().cpu())
            
            # xs = 0
            # xv = 0 

            # for vx, sx in zip(video_emb_dict["sub_clip"], text_emb_dict["sub_sentence"]):
            #     xv += vx
            #     xs += sx        

            # emN1 = (video_emb_dict["clip_emb"] + xv / 4) / 4
            # emN2 = (text_emb_dict["sentence_emb"] + xs/4) / 4

            # clip_emb_list1.extend(emN1.detach().cpu())
            # sent_emb_list1.extend(emN2.detach().cpu())

            clip_emb_list.extend(video_emb_dict["local_emb_a"].detach().cpu())
            sent_emb_list.extend(text_emb_dict["local_emb_b"].detach().cpu())

            # logging
            if step % 10 == 0:
                self.logger.info(
                    f"Test [{step}/{max_step}] Loss {loss.item():.4f}")

        vid_emb_list = torch.stack(vid_emb_list, 0)
        par_emb_list = torch.stack(par_emb_list, 0)

        clip_emb_list = torch.stack(clip_emb_list, 0)
        sent_emb_list = torch.stack(sent_emb_list, 0)



        # clip_emb_list1 = torch.stack(clip_emb_list1, 0)
        # sent_emb_list1 = torch.stack(sent_emb_list1, 0)
        # video text retrieval
        vid_emb_list = F.normalize(vid_emb_list).numpy()
        par_emb_list = F.normalize(par_emb_list).numpy()

        # print(len(clip_emb_list), clip_emb_list[0])
        clip_emb_list = F.normalize(clip_emb_list).numpy()
        sent_emb_list = F.normalize(sent_emb_list).numpy()

        #====== Save embeddings ====
        if self.save_emb_path != "":
            # save unnormalized embeddings
            os.makedirs(self.save_emb_path, exist_ok=True)
            # print(self.path_ckpt)
            # import pdb;pdb.set_trace()
            gp_obj_file = os.path.join(self.save_emb_path, 'text_to_act_app_H16.pickle')
            test_data = {}
            test_data["video_org"] = save_clip_vec
            test_data["text_org"] = save_sent_vec
            test_data["text"] = save_sent_cap
            test_data["key"] = save_key
            test_data["text_emb"] = clip_emb_list
            test_data["video_emb"] = sent_emb_list
            with open(gp_obj_file, 'wb') as f:
                pk.dump(test_data, f, pk.HIGHEST_PROTOCOL)
            self.logger.info(f"Saved embeddings to {gp_obj_file}\n")

        # clip_emb_list1 = F.normalize(clip_emb_list1).numpy()
        # sent_emb_list1 = F.normalize(sent_emb_list1).numpy()

        # calculate total loss and feed meters
        loss_total /= num_steps
        # print("=="*20, loss_total, loss)
        forward_time_total /= num_steps
        self.metrics.update_meter(MetersConst.VAL_LOSS_CONTRASTIVE, loss_total)

        # calculate video-paragraph retrieval and print output table
        self.logger.info(compute_retrieval.VALHEADER)

        v2p_res, p2v_res, sum_vp_at_1, str_vp = compute_retrieval.compute_scores(
            vid_emb_list, par_emb_list, "vid_emb", "par_emb", print_fn=self.logger.info)

        c2s_res, s2c_res, sum_cs_at_1, str_cs = compute_retrieval.compute_scores(
                clip_emb_list, sent_emb_list, "clip_emb", "sent_emb", print_fn=self.logger.info)

        # c2s_res1, s2c_res1, sum_cs_at_11, str_cs1 = compute_retrieval.compute_scores(
        #         clip_emb_list1, sent_emb_list1, "clip_emb", "sent_emb", print_fn=self.logger.info)


        # feed retrieval results to meters
        for modality, dict_ret in zip(MetersConst.RET_MODALITIES, [v2p_res, p2v_res, c2s_res, s2c_res]):
            if dict_ret is None:
                continue
            # iterate over result keys
            for metric in MetersConst.RET_METRICS:
                # feed averagemeters
                logger_class = "test_ret"
                if metric == "r1":
                    logger_class = "test_base"
                self.metrics.update_meter(f"{logger_class}/{modality}-{metric}", dict_ret[metric])

        # print some more details about the retrieval (time, number of datapoints)
        self.logger.info(
            f"Loss {loss_total:.5f} "
            f"Retrieval: {str_vp}{str_cs}total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s")
