import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, y_hat, y):
        return self._dice_loss(y_hat, y)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


class DiceRMSELoss(nn.Module):
    def __init__(self, dice_ratio=0.6, rmse_ratio=0.4):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.rmse_loss = RMSELoss()
        self.dice_ratio = dice_ratio
        self.rmse_ratio = rmse_ratio

    def forward(self, y_hat, y):
        _dice = self.dice_loss(y_hat, y)
        _dice = self.dice_ratio * _dice
        _rmse = self.rmse_loss(y_hat, y)
        _rmse = self.rmse_ratio * _rmse
        return _dice + _rmse


class DiceCoefLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _dice_coef(self, y_true, y_pred):
        smooth = 1.0
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        a = (2. * intersection + smooth)
        b = (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return a / b

    def forward(self, y_hat, y):
        return - self._dice_coef(y, y_hat)
