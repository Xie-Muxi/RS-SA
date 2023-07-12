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
    #TODO: Overwrite
    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):
        # Reshape inputs and targets
        num_classes = inputs.shape[1]  # Assuming inputs has shape [batch_size, num_classes, height, width]

        print(num_classes)

        print('----------- before -----------')
        print(inputs.shape)
        print(targets.shape)
        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # Reshape inputs to [batch_size * height * width, num_classes]
        targets = targets.view(-1).long()  # Reshape targets to [batch_size * height * width]
        print('----------- after -----------')
        print(inputs.shape)
        print(targets.shape)

        # Compute the focal loss
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = alpha * (1-pt)**gamma * CE_loss
        return F_loss.mean()


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    # #TODO: Overwrite
    # def forward(self, inputs, targets, smooth=1):
        
    #     inputs = F.softmax(inputs, dim=1)
    #     inputs = torch.clamp(inputs, min=0, max=1)
    #     inputs = inputs.view(-1)
    #     targets = targets.view(-1)

    #     intersection = (inputs * targets).sum()
    #     dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    #     return 1 - dice

#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return 1 - dice

    def forward(self, inputs, targets, smooth=1):
        # Reshape inputs and targets
        num_classes = inputs.shape[1]  # Assuming inputs has shape [batch_size, num_classes, height, width]

        print(num_classes)
        print('----------- before -----------')
        print(inputs.shape)
        print(targets.shape)

        inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # Reshape inputs to [batch_size * height * width, num_classes]
        targets = targets.view(-1).long()  # Reshape targets to [batch_size * height * width]

        print('----------- after -----------')
        print(inputs.shape)
        print(targets.shape)

        # Compute the dice loss for each class separately
        dice_loss = 0
        for i in range(num_classes):
            input_i = inputs[:, i]
            target_i = (targets == i).float()
            intersection = (input_i * target_i).sum()
            dice_loss_i = 1 - (2. * intersection + smooth) / (input_i.sum() + target_i.sum() + smooth)
            dice_loss += dice_loss_i
        return dice_loss / num_classes