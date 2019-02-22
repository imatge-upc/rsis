import torch
from .hungarian import softIoU, MaskedNLL, StableBalancedMaskedBCE
import torch.nn as nn


class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None, gamma=0.0):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight, self.gamma).view(-1,1)
       
        return costs


class MaskedMSELoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedMSELoss, self).__init__()
        self.balance_weight = balance_weight

    def forward(self, y_true, y_pred):
        costs = nn.MSELoss()(y_true, y_pred).view(-1, 1)
        return costs


class MaskedBoxLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedBoxLoss, self).__init__()
        self.balance_weight = balance_weight

    def forward(self, y_true, y_pred):
        costs = - torch.nn.functional.log_softmax(y_pred)*y_true

        costs = torch.sum(costs, dim=-1)
        return costs


class MaskedBCELoss(nn.Module):

    def __init__(self,mask_mode=False, gamma=0):
        super(MaskedBCELoss,self).__init__()
        self.mask_mode = mask_mode
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.mask_mode, self.gamma).view(-1,1)
        
        return costs


class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()

    def forward(self, y_true, y_pred):
        costs = softIoU(y_true,y_pred).view(-1,1)
        #if sw is not None:
            #costs = torch.masked_select(costs,sw.byte())
        return costs
