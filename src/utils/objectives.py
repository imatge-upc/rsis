import torch
from hungarian import softIoU, MaskedNLL, StableBalancedMaskedBCE, MonotonicMaskedBCE, dice
import torch.nn as nn

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        costs = self.nll_loss(torch.nn.functional.log_softmax(inputs), targets)
        return costs

class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred, sw):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1)
        # costs = torch.mean(torch.masked_select(costs,sw.byte()))
        costs = torch.masked_select(costs,sw.byte())
        return costs

class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred,sw):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        costs = torch.masked_select(costs,sw.byte())
        return costs

class StableBalancedMaskedBCELoss(nn.Module):
    def __init__(self):
        super(StableBalancedMaskedBCELoss,self).__init__()
    def forward(self, y_true, y_pred, sw):
        costs = StableBalancedMaskedBCE(y_true,y_pred).view(-1,1)
        costs = torch.mean(torch.masked_select(costs, sw.byte()))
        return costs

class StableMaskedBCELoss(nn.Module):
    def __init__(self):
        super(StableMaskedBCELoss,self).__init__()
    def forward(self, y_true, y_pred, sw):
        costs = StableMaskedBCE(y_true,y_pred).view(-1,1)
        costs = torch.mean(torch.masked_select(costs, sw.byte()))
class MonMaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MonMaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred,sw):
        costs = MonotonicMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        costs = torch.masked_select(costs,sw.byte())
        return costs

class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred, sw):
        costs = softIoU(y_true,y_pred).view(-1,1)
        costs = torch.mean(torch.masked_select(costs,sw.byte()))
        return costs

class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss,self).__init__()
    def forward(self, y_true, y_pred, sw):
        costs = dice(y_true,y_pred).view(-1,1)
        costs = torch.mean(torch.masked_select(costs,sw.byte()))
        return costs