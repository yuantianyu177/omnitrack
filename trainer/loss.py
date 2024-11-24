import torch
import torch.nn.functional as F


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile=1):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1, keepdim=True)
        loss_at_quantile = torch.quantile(sum_loss, quantile)
        quantile_mask = (sum_loss < (loss_at_quantile+1e-7)).squeeze(-1)
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])
        
def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss