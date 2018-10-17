import torch
from torch.autograd import Variable
import torch.nn as nn
from munkres import Munkres
import numpy as np
import time

torch.manual_seed(0)


def MaskedNLL(target, probs, balance_weights=None):

    log_probs = torch.log(probs+1e-6)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()


def StableBalancedMaskedBCE(target, out, mask_mode=True):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """

    if mask_mode:
        th_pred = (torch.sigmoid(out).detach() > 0.5).float()
        mask = target + th_pred - th_pred*target

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()

    if mask_mode:
        loss_values = loss_values * mask
        loss_values = loss_values.sum(dim=-1) / ((mask.sum(dim=-1)) + 1e-6)
    else:
        loss_values = torch.mean(loss_values, dim=-1)

    return loss_values.squeeze()


def softIoU(target, out, e=1e-6):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """

    # clamp values to avoid nan loss
    #out = torch.clamp(out,min=e,max=1.0-e)
    #target = torch.clamp(target,min=e,max=1.0-e)

    num = (out*target).sum(1,True)
    den = (out+target-out*target).sum(1,True) + e
    iou = num / den
    # set iou to 0 for masks out of range
    # this way they will never be picked for hungarian matching
    cost = (1 - iou)

    return cost.squeeze()

def match(masks, classes, overlaps, boxes):
    """
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        classes - list containing [true_classes, pred_classes] with shape [batch_size,T,]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        t_class_cpu - [batch_size,T,] permuted ground truth classes
        permute_indices - permutation indices used to sort the above
    """

    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks
    t_class, p_class = classes

    # get true mask values to cpu as well
    t_mask_cpu = (t_mask.data).cpu().numpy()
    t_class_cpu = (t_class.data).cpu().numpy()
    t_box_cpu = (boxes.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0), t_mask.size(1)),dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample, column] = row

        # sort ground according to permutation
        t_mask_cpu[sample] = t_mask_cpu[sample, permute_indices[sample], :]
        t_box_cpu[sample] = t_box_cpu[sample, permute_indices[sample], :]
        t_class_cpu[sample] = t_class_cpu[sample, permute_indices[sample]]
    return t_mask_cpu, t_class_cpu, t_box_cpu, permute_indices
