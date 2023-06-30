# losses.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# ALPHA = 0.8
# GAMMA = 2


# class FocalLoss(nn.Module):

#     def __init__(self, weight=None, size_average=True):
#         super().__init__()

#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

#         return focal_loss


# class DiceLoss(nn.Module):

#     def __init__(self, weight=None, size_average=True):
#         super().__init__()

#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice


# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, num_classes, weight=None, size_average=True):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        # convert targets to one-hot format to match the shape of inputs
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets_one_hot = targets_one_hot.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets_one_hot, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, num_classes, weight=None, size_average=True):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)

        # convert targets to one-hot format to match the shape of inputs
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets_one_hot = targets_one_hot.view(-1)

        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets_one_hot.sum() + smooth)

        return 1 - dice
