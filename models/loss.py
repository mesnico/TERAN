import torch
from torch import nn as nn
from torch.nn import functional as F
from .utils import l2norm


def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Contrastive(nn.Module):
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class AlignmentContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, aggregation='sum-max-sentences', return_similarity_mat=False):
        super(AlignmentContrastiveLoss, self).__init__(margin, measure, max_violation)
        self.aggregation = aggregation
        self.return_similarity_mat = return_similarity_mat

    def forward(self, im_set, s_seq, im_len, s_len):
        # im_set = im_set.permute(1, 0, 2)    # B x S_im x dim
        # s_seq = s_seq.permute(1, 0, 2)     # B x S_s x dim

        # do not consider cls and eos tokens
        im_set = im_set[:, 1:, :]
        s_seq = s_seq[:, 1:-2, :]
        im_len = [l - 1 for l in im_len]
        s_len = [l - 3 for l in s_len]

        im_set_batch = im_set.size(0)
        im_set_len = im_set.size(1)
        s_seq_batch = s_seq.size(0)
        s_seq_len = s_seq.size(1)

        im_set = im_set.unsqueeze(1).expand(-1, s_seq_batch, -1, -1)  # B x B x S_im x dim
        s_seq = s_seq.unsqueeze(0).expand(im_set_batch, -1, -1, -1) # B x B x S_s x dim
        alignments = torch.matmul(im_set, s_seq.permute(0, 1, 3, 2))  # B x B x S_im x S_s
        # alignments = F.relu(alignments)

        # compute mask for the alignments tensor
        im_len_mask = torch.zeros(im_set_batch, im_set_len).bool()
        im_len_mask = im_len_mask.to(im_set.device)
        for im, l in zip(im_len_mask, im_len):
            im[l:] = True
        im_len_mask = im_len_mask.unsqueeze(2).unsqueeze(1).expand(-1, s_seq_batch, -1, s_seq_len)

        s_len_mask = torch.zeros(s_seq_batch, s_seq_len).bool()
        s_len_mask = s_len_mask.to(im_set.device)
        for sm, l in zip(s_len_mask, s_len):
            sm[l:] = True
        s_len_mask = s_len_mask.unsqueeze(1).unsqueeze(0).expand(im_set_batch, -1, im_set_len, -1)

        alignment_mask = im_len_mask | s_len_mask
        alignments.masked_fill_(alignment_mask, value=0)
        # alignments = F.relu(alignments)
        # alignments = F.normalize(alignments,p=2, dim=2)

        if self.aggregation == 'sum':
            aggr_similarity = alignments.sum(dim=(2,3))
        elif self.aggregation == 'mean':
            aggr_similarity = alignments.mean(dim=(2,3))
        elif self.aggregation == 'MrSw':
            aggr_similarity = alignments.max(2)[0].sum(2)
        elif self.aggregation == 'MrAVGw':
            aggr_similarity = alignments.max(2)[0].sum(2)
            expanded_len = torch.FloatTensor(s_len).to(alignments.device).unsqueeze(0).expand(len(im_len), -1)
            aggr_similarity /= expanded_len
        elif self.aggregation == 'symm':
            im = alignments.max(2)[0].sum(2)
            s = alignments.max(3)[0].sum(2)
            aggr_similarity = im + s
        elif self.aggregation == 'MwSr':
            aggr_similarity = alignments.max(3)[0].sum(2)
        elif self.aggregation == 'scan-sentences':
            norm_alignments = F.relu(alignments)
            norm_alignments = F.normalize(norm_alignments,p=2, dim=2)
            weights = norm_alignments.masked_fill(alignment_mask, value=float('-inf'))
            weights = torch.softmax(weights, dim=3)

            weights = weights.unsqueeze(3)  # B x B x im x 1 x s
            s_seq_ext = s_seq.unsqueeze(2).expand(-1, -1, im_set_len, -1, -1)
            att_vector = torch.matmul(weights, s_seq_ext)  # B x B x im x 1 x dim
            att_vector = att_vector.squeeze(3)
            new_alignments = F.cosine_similarity(im_set, att_vector, dim=3)  # B x B x im
            new_alignments.masked_fill_(im_len_mask[:, :, :, 0], value=0)

            aggr_similarity = new_alignments.sum(2)

        if self.return_similarity_mat:
            return aggr_similarity
        else:
            loss = self.compute_contrastive_loss(aggr_similarity)
            return loss


class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'cosine':
            self.sim = cosine_sim
        elif measure == 'dot':
            self.sim = dot_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        return self.compute_contrastive_loss(scores)


class PermInvMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # @staticmethod
    # def batched_cosine_sim(im, s):
    #     """Cosine similarity between all the image and sentence pairs
    #     """
    #     im = F.normalize(im, p=2, dim=2)
    #     s = F.normalize(s, p=2, dim=2)
    #     return im.mm(s.permute(0, 2, 1))

    def forward(self, im, s):
        dist_matrix = torch.cdist(im, s, p=2)
        row_sum = F.softmin(dist_matrix, dim=2).max(dim=2)[0].sum(dim=1)
        col_sum = F.softmin(dist_matrix, dim=1).max(dim=1)[0].sum(dim=1)
        loss = 2*torch.Tensor([dist_matrix.shape[1]]).to(im.device) - row_sum - col_sum
        loss = loss.mean()
        return loss
