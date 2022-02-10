from torch import nn
import torch
import torch.nn.functional as F
import time 
import numpy as np

def cosine_sim(emb1, emb2):
    """compute cosine similarity of two embeddings
    Args:
        emb1 
        emb2 
    Returns:
        float: cosine similarity between (-1, 1)
    """    
    return emb1.mm(emb2.t())

class MaxMargin_coot(nn.Module):
    """Regular Contrastive Loss between 2 groups of embeddings
    inputs shape (batch, embed_dim)
    Ref: COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning, NeurIPS 2020
    """

    def __init__(self, use_cuda: bool, margin: float = 0.1):
        super(ContrastiveLoss_coot, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.use_cuda = use_cuda

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        if self.use_cuda:
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
    
    
class CrossCLR_onlyIntraModality(nn.Module):
    """
    CrossCLR Loss between 2 groups of embeddings - Only Intra Modality alignment
    ICCV 2021
    """

    def __init__(self, temperature=0.03, negative_weight=0.8, logger = None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') 
        self.temperature = temperature 
        self.logger = logger
        self.negative_w = negative_weight # Weight of negative samples logits.


    def compute_loss(self, logits, mask):
        return - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def forward(self, video_features, text_features):
        """
        Inputs shape (batch, embed_dim)
        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)
        Returns:
        """
        batch_size = video_features.shape[0]

        # Normalize features 
        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)
        
        # Inter-modality alignment
        logits_per_vid = video_features @ text_features.t()
        logits_per_text = text_features @ video_features.t()

        # Intra-modality alignment
        logits_clstr_vid = video_features @ video_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        logits_per_vid /= self.temperature 
        logits_per_text /= self.temperature 
        logits_clstr_vid /= self.temperature 
        logits_clstr_txt /= self.temperature 

        positive_mask = self._get_positive_mask( video_features.shape[0])
        negatives_vid = logits_clstr_vid * positive_mask
        negatives_txt = logits_clstr_txt * positive_mask

        vid_logits = torch.cat([logits_per_vid, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([logits_per_text, self.negative_w * negatives_txt], dim=1)

        diag = np.eye(batch_size)
        mask_vid = torch.from_numpy((diag)).cuda()
        mask_txt = torch.from_numpy((diag)).cuda()

        mask_neg_v = torch.zeros_like(negatives_vid)
        mask_neg_t = torch.zeros_like(negatives_txt)
        mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
        mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

        loss_i = self.compute_loss(vid_logits, mask_v)
        loss_t = self.compute_loss(txt_logits, mask_t)

        return ((loss_i.mean() + loss_t.mean())  )  / 2


class CrossCLR_noq(nn.Module):
    """
    CrossCLR Loss between 2 groups of embeddings
    """

    def __init__(self, temperature=0.03, temperature_weights=0.0035, negative_weight=0.8, 
        score_thrshold=0.7, logger = None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none') #torch.nn.CrossEntropyLoss()
        self.temperature = temperature #
        self.logger = logger
    

        self.score_thrshold = score_thrshold
        self.temp_w = temperature_weights # Temperature for scaling weights. 
        self.negative_w = negative_weight # Weight of negative scores.
        self.logger.info("==="*30)
        self.logger.info("Temp:{}, TempW:{}, NegW:{}, Sthrsh:{}".format(self.temperature,self.temp_w,self.negative_w,self.score_thrshold))
        self.logger.info("==="*30)
        # create the queue


    def compute_loss(self, logits, mask):

        loss = - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )
        return loss #loss.mean()

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def _get_positive_mask_bank(self, k, batch_size, ptr):
        diag = np.eye(batch_size)
        mask = torch.from_numpy((diag))
        # mask = (1 - mask)

        diag_bank = np.ones((batch_size, k))
        mask_bank = torch.from_numpy((diag_bank))

        if (ptr+batch_size) > k:
            qptr_end = k
            inp_feat_k = batch_size - (ptr+batch_size - k)
            mask_bank[:, ptr:] -= mask[:,:inp_feat_k]
        else:
            mask_bank[:, ptr:ptr+batch_size] -= mask

        return mask_bank.cuda(non_blocking=True)


    def forward(self, video_features, text_features, input_vid=None, input_txt=None):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """


        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)
        input_vid = input_vid.cuda()
        input_txt = input_txt.cuda()


        logits_per_image = video_features @ text_features.t()
        logits_per_text = text_features @ video_features.t()

        logits_clstr_vid = video_features @ video_features.t()
        logits_clstr_txt = text_features @ text_features.t()


        logits_per_image /= self.temperature 
        logits_per_text /= self.temperature 
        logits_clstr_vid /= self.temperature 
        logits_clstr_txt /= self.temperature 

        positive_mask = self._get_positive_mask( video_features.shape[0])
        sim_scores_vid = (input_vid @ input_vid.t()) * positive_mask
        sim_scores_txt= (input_txt @ input_txt.t()) * positive_mask
            

        avg_sim_vid = torch.mean(sim_scores_vid,dim=1)
        avg_sim_txt = torch.mean(sim_scores_txt,dim=1)
        

        sorted_vid, indices_vid = torch.sort(avg_sim_vid)
        sorted_txt, indices_txt = torch.sort(avg_sim_txt)
        sorted_vid = sorted_vid / sorted_vid.max(dim=-1, keepdim=True)[0]
        sorted_txt = sorted_txt / sorted_txt.max(dim=-1, keepdim=True)[0]


        # ======================================================
        # Find index of influential samples and remove them from negative set
        indices_vid_thrsh = indices_vid[sorted_vid<self.score_thrshold]
        indices_txt_thrsh = indices_txt[sorted_txt<self.score_thrshold]

        labels = torch.arange(video_features.shape[0]).cuda()

        logits_clstr_vid = logits_clstr_vid * positive_mask
        logits_clstr_txt = logits_clstr_txt * positive_mask

        negatives_vid = logits_clstr_vid[:, indices_vid_thrsh]
        negatives_txt = logits_clstr_txt[:, indices_txt_thrsh]


        batch_size = input_vid.shape[0]
        labels_prune = torch.arange(batch_size).cuda()

        
        vid_logits_prune = logits_per_image
        txt_logits_prune = logits_per_text

        prune_pos = 1
        if prune_pos:
            sorted_vid2, indices_vid2 = torch.sort(avg_sim_vid)
            sorted_txt2, indices_txt2 = torch.sort(avg_sim_txt)
            sorted_vid2 = sorted_vid2 / sorted_vid2.max(dim=-1, keepdim=True)[0]
            sorted_txt2 = sorted_txt2 / sorted_txt2.max(dim=-1, keepdim=True)[0]
            indices_vid_thrsh2 = indices_vid2[sorted_vid2>self.score_thrshold]
            indices_txt_thrsh2 = indices_txt2[sorted_txt2>self.score_thrshold]

            diag = np.eye(batch_size)
            mask = torch.from_numpy((diag)).cuda()
            mask_prune_pos_vid = torch.ones_like(logits_per_image)
            mask_prune_pos_txt = torch.ones_like(logits_per_text)

            mask_prune_pos_vid[:,indices_vid_thrsh2] = 0
            mask_prune_pos_txt[:,indices_txt_thrsh2] = 0

            for i in range(batch_size): 
                if mask_prune_pos_vid[i,i]==0: mask_prune_pos_vid[i,i]=1
                if mask_prune_pos_txt[i,i]==0: mask_prune_pos_txt[i,i]=1

            vid_logits_prune = logits_per_image * mask_prune_pos_vid
            txt_logits_prune = logits_per_text * mask_prune_pos_txt


        vid_logits = torch.cat([vid_logits_prune, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([txt_logits_prune, self.negative_w * negatives_txt], dim=1)


        diag = np.eye(batch_size)
        mask_vid = torch.from_numpy((diag)).cuda()
        mask_txt = torch.from_numpy((diag)).cuda()

        multi_pos = 0
        num_p = 5
        mp_score = 0.15
        if multi_pos:
            positive_mask = self._get_positive_mask( video_features.shape[0])
            sim_mask_vid = (input_vid @ input_vid.t()) * positive_mask
            sim_mask_txt= (input_txt @ input_txt.t()) * positive_mask
            _, topkidx_vid = torch.topk(sim_mask_vid, num_p, dim=1)
            topk_onehot_vid = torch.zeros_like(sim_mask_vid)
            topk_onehot_vid.scatter_(1, topkidx_vid, 1)
            mask_vid[topk_onehot_vid.bool()] = mp_score

            _, topkidx_txt = torch.topk(sim_mask_txt, num_p, dim=1)
            topk_onehot_txt = torch.zeros_like(sim_mask_txt)
            topk_onehot_txt.scatter_(1, topkidx_txt, 1)
            mask_txt[topk_onehot_txt.bool()] = mp_score

        mask_neg_v = torch.zeros_like(negatives_vid)
        mask_neg_t = torch.zeros_like(negatives_txt)
        mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
        mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

        loss_i = self.compute_loss(vid_logits, mask_v)
        loss_t = self.compute_loss(txt_logits, mask_t)


        w_i =  ((avg_sim_vid/sum(avg_sim_vid)))
        w_t = ((avg_sim_txt/sum(avg_sim_txt)))
        loss_i = loss_i * torch.exp(w_i/ self.temp_w)
        loss_t = loss_t * torch.exp(w_t/ self.temp_w)


        loss_i = sum(loss_i) / (sum(torch.exp(w_i/ self.temp_w)))
        loss_t = sum(loss_t) / (sum(torch.exp(w_t/ self.temp_w)))


        loss = ((loss_i + loss_t)  )  / 2

        return loss