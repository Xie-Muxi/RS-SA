import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

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
    
    # modified by Xie-Muxi, use cross_entropy instead of binary_cross_entropy to support multi-class
    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # remove sigmoid and flatten
        inputs = inputs.view(-1, inputs.shape[-1])  # shape: (batch_size * height * width, num_classes)
        targets = targets.view(-1)  # shape: (batch_size * height * width)

        CE = F.cross_entropy(inputs, targets, reduction='mean')
        CE_EXP = torch.exp(-CE)
        focal_loss = alpha * (1 - CE_EXP)**gamma * CE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice

    # def forward(self, inputs, targets, smooth=1):
    #     # Convert targets to one-hot encoding
    #     targets_onehot = torch.zeros_like(inputs)
    #     targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

    #     # Compute the dice loss for each class separately and average them
    #     dice_loss = 0
    #     for i in range(inputs.size(1)):
    #         inputs_i = inputs[:, i]
    #         targets_i = targets_onehot[:, i]

    #         inputs_i = torch.sigmoid(inputs_i)
    #         inputs_i = torch.clamp(inputs_i, min=0, max=1)

    #         intersection = (inputs_i * targets_i).sum()
    #         dice_i = (2. * intersection + smooth) / (inputs_i.sum() + targets_i.sum() + smooth)

    #         dice_loss += dice_i

    #     return 1 - dice_loss / inputs.size(1)

    def forward(self, inputs, targets, smooth=1):
        inputs = F.softmax(inputs, dim=1)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
