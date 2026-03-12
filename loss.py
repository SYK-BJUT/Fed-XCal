from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#L_Scl
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class SimplifiedSupConLoss(nn.Module):


    def __init__(self, temperature=0.07):
        super(SimplifiedSupConLoss, self).__init__()
        self.supcon_loss = SupConLoss(temperature=temperature)

    def forward(self, features, labels):
        return self.supcon_loss(features, labels)
#L_Arc
class ArcFaceLoss(nn.Module):
    """ArcFaceLoss Deng, J. et al. ArcFace: Additive angular margin loss for deep face recognition. IEEE Trans. Pattern Anal. Mach. Intell. 44, 5962–5979 (2022)."""

    def __init__(self, num_classes, feat_dim, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m


        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)


        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m



    def forward(self, x, labels):

        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)


        cos_theta = torch.mm(x_norm, w_norm.t())
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)


        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))


        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m


        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)


        batch_size = labels.size(0)
        one_hot = torch.zeros(cos_theta.size(), device=x.device, dtype=cos_theta.dtype)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)


        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)


        output *= self.s


        return F.cross_entropy(output, labels)
#L_Cen
class CenterLoss(nn.Module):
    """Wen, Y., Zhang, K., Li, Z. & Qiao, Y. A discriminative feature learning approach

    for deep face recognition. In Proc. Eur. Conf. Comput. Vis. 499–515 (2016)."""

    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha


        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)



    def forward(self, x, labels):

        batch_size = x.size(0)


        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)


        classes = torch.arange(self.num_classes).long()
        if x.is_cuda:
            classes = classes.cuda()

        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))


        dist = distmat * mask.float()
        center_loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return center_loss

#Uncertainty weighting
class UncertaintyWeighting(nn.Module):
    """Kendall, A., Gal, Y. & Cipolla, R. Multi-task learning using uncertainty to weigh

    losses for scene geometry and semantics. In Proc. IEEE/CVF Conf. Comput. Vis.

    Pattern Recognit. 7482–7491 (2018)."""
    def __init__(self, num_tasks=6, initial_weights=None):
        super(UncertaintyWeighting, self).__init__()

        if initial_weights is None:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        else:

            initial_weights_tensor = torch.tensor(initial_weights, dtype=torch.float32)
            log_vars_init = -torch.log(initial_weights_tensor)
            self.log_vars = nn.Parameter(log_vars_init)
            print(f"Uncertainty Weighting初始化: num_tasks={num_tasks}, 自定义初始化")
            print(
                f"初始权重: CE={initial_weights[0]:.4f}, SupCon={initial_weights[1]:.4f}, MSE={initial_weights[2]:.4f}, FedProx={initial_weights[3]:.4f}, ArcFace={initial_weights[4]:.4f}, Center={initial_weights[5]:.4f}")

    def forward(self, losses_dict):

        loss_names_order = ['ce', 'supcon', 'mse', 'fedprox', 'arcface', 'center']

        active_loss_names = [name for name in loss_names_order if name in losses_dict]
        losses = [losses_dict[name] for name in active_loss_names]

        if len(losses) == 0:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor([])

        weighted_losses = []

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])

            weighted_loss = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted_loss)

        total_weighted_loss = sum(weighted_losses)

        with torch.no_grad():
            current_weights = torch.exp(-self.log_vars[:len(losses)])

        return total_weighted_loss, current_weights

    def get_current_weights(self):

        return torch.exp(-self.log_vars).detach()

    def get_log_vars(self):

        return self.log_vars.detach()