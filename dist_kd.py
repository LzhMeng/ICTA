import torch
import torch.nn as nn


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
    # return nn.MSELoss()(a,b)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)

def compute_rank_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """
    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

    att = att.sort(dim=1)[1]
    grad_att = grad_att.sort(dim=1)[1]
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation

def inter_class_relation0(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()
def inter_class_relation(y_s, y_t):
    # return 1 - pearson_correlation(y_s, y_t).mean()
    return compute_rank_correlation(y_s, y_t)


def intra_class_relation(y_s, y_t):
    # return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
    return inter_class_relation0(y_s.transpose(0, 1), y_t.transpose(0, 1))
    # return compute_rank_correlation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss