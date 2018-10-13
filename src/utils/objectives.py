import torch
from .hungarian import softIoU, MaskedNLL, StableBalancedMaskedBCE
import torch.nn as nn


class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight

    def forward(self, y_true, y_pred, sw=None):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1)
        # costs = torch.mean(torch.masked_select(costs,sw.byte()))
        if sw is not None:
            costs = torch.masked_select(costs,sw.byte())
        return costs


class MaskedBoxLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedBoxLoss, self).__init__()
        self.balance_weight = balance_weight

    def forward(self, y_true, y_pred, sw=None):
        costs = - torch.nn.functional.log_softmax(y_pred)*y_true
        # costs = torch.mean(torch.masked_select(costs,sw.byte()))
        costs = torch.sum(costs, dim=-1)
        if sw is not None:
            costs = torch.masked_select(costs, sw.byte())
        return costs


class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight

    def forward(self, y_true, y_pred, sw=None):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        if sw is not None:
            costs = torch.masked_select(costs,sw.byte())
        return costs


class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()

    def forward(self, y_true, y_pred, sw=None):
        costs = softIoU(y_true,y_pred).view(-1,1)
        if sw is not None:
            costs = torch.masked_select(costs,sw.byte())
        return costs
