from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from classes.MNet_class import MedicalVolumeDataset_3D, MNet3D

def hybrid_loss(pred, target):
    ce = F.cross_entropy(pred, target)
    pred_prob = F.softmax(pred, dim=1)
    dice = 1 - dice_score_3d(pred_prob, target)
    return ce + dice

def dice_score_3d(pred, target):
    smooth = 1e-6
    pred = pred.argmax(1)
    scores = []
    for class_idx in range(3):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append((2.*intersection + smooth)/(union + smooth))
    return torch.mean(torch.tensor(scores))

# Data preparation
volume_indices = list(range(2))  # Replace with full range for real training
train_volumes, test_volumes = train_test_split(volume_indices, test_size=0.2, random_state=42)
train_dataset = MedicalVolumeDataset_3D("data/LITS17", train_volumes)
test_dataset = MedicalVolumeDataset_3D("data/LITS17", test_volumes)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

# Model, optimizer, device
model = MNet3D(n_channels=1, n_classes=3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
best_dice = 0
for epoch in range(500):
    print(f"\nEpoch {epoch+1}/500")
    model.train()
    train_losses = []
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = hybrid_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    print(f" Train loss: {sum(train_losses)/len(train_losses):.4f}")

    # Validation
    model.eval()
    class_dice = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1)
            outputs = model(inputs)
            batch_score = dice_score_3d(outputs, targets)
            for i in range(3):
                class_dice[i] += batch_score[i]
    avg_dice = [c/len(test_loader) for c in class_dice]
    mean_dice = sum(avg_dice) / len(avg_dice)
    print(f" Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}, Mean: {mean_dice:.4f}")
    if avg_dice[2] > best_dice:
        best_dice = avg_dice[2]
        torch.save(model.state_dict(), "best_mnet_model.pth")
        print(f"\nBest Tumor Dice: {best_dice*100:.2f}%")
