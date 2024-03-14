import pdb
import pickle

import torch
import torch.nn.functional as F
from sentence_transformers import util
from torch import nn


class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x / self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t() / self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return -loss_i - loss_j


class EgoNCE(nn.Module):
    def __init__(self, temperature=0.05, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n):
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        elif self.verb:
            mask = mask_v + mask_diag
        else:
            mask = mask_diag

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x / self.temperature, dim=1)
        j_sm = F.softmax(x.t() / self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1))
        loss_j = jdiag.sum() / len(jdiag)
        return -loss_i - loss_j


class MaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class AdaptiveMaxMarginRankingLoss(nn.Module):
    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


class AudioTextContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sim_a2t, sim_t2a, sim_targets=None):
        if sim_targets is None:
            sim_targets = torch.zeros(sim_a2t.size()).to(sim_a2t.device)
            sim_targets.fill_diagonal_(1)

        loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1) * sim_targets, dim=1).mean()

        loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1) * sim_targets, dim=1).mean()

        loss_atc = (loss_a2t + loss_t2a) / 2
        return loss_atc


class NTXent(nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):
        n = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        # mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        # mask_diag = mask.diag()
        # mask_diag = torch.diag_embed(mask_diag)
        # mask = mask ^ mask_diag
        #
        # a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        # t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()

        t2a_loss = -self.loss(t2a).mean()
        a2t_loss = -self.loss(a2t).mean()

        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss
