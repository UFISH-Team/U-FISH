import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        return (input_tensor == 1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        target = self._one_hot_encoder(target)
        return self._dice_loss(inputs, target)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
