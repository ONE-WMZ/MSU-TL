import torch.nn as nn


# ! huber_loss
def huber_loss(pred, target, mask):
    SmoothL1Loss = nn.SmoothL1Loss(reduction='none')
    loss = SmoothL1Loss(pred, target)
    if mask is not None:
        # app mask
        loss = loss * mask
        # Normalization
        total_loss = loss.sum() / mask.sum().clamp(min=1e-6)
    else:
        # mean
        total_loss = loss.mean()
    return total_loss