import torch
import torch.nn.functional as F

# Hybrid Loss (CE + Dice)
def hybrid_loss(pred, target):
    ce = F.cross_entropy(pred, target)
    pred_prob = F.softmax(pred, dim=1)
    dice = 1 - dice_score_3d(pred_prob, target)
    return ce + dice

# 3D Dice Calculation
def dice_score_3d(pred, target):
    smooth = 1e-6
    pred = pred.argmax(1) #converts to 0,1, 2
    scores = []
    for class_idx in range(3):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append((2.*intersection + smooth)/(union + smooth))
    return torch.mean(torch.tensor(scores))


# 3D Dice Calculation for each label
def dice_score_3d_perclass(pred, target, num_classes=3):
    """Returns list of Dice scores, one per class."""
    smooth = 1e-6
    pred = pred.argmax(1)  # (B, D, H, W)
    scores = []
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        scores.append(dice.item())
    return scores





