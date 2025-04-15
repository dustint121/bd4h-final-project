from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from classes.UNet2D_class import UNet2D, MedicalSliceDataset_2D
from tqdm import tqdm


def dice_score(pred, target):
    smooth = 1e-6
    pred = F.softmax(pred, dim=1)
    scores = []
    # Calculate Dice for each class (background, organ, tumor)
    for class_idx in range(3):
        pred_mask = pred[:, class_idx]  # Get probability maps for this class
        target_mask = (target == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append((2. * intersection + smooth) / (union + smooth))
    
    return torch.mean(torch.stack(scores))  # Return mean Dice across classes


# Additional helper function for class-wise Dice reporting
def dice_score_per_class(pred, target):
    smooth = 1e-6
    pred = F.softmax(pred, dim=1).argmax(1)
    scores = []
    for class_idx in range(3):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append((2.*intersection + smooth)/(union + smooth))
    return scores


def hybrid_loss(pred, target):
    # Cross Entropy
    ce = F.cross_entropy(pred, target)
    
    # Dice Loss (1 - mean Dice score)
    dice_loss = 1 - dice_score(pred, target)
    
    return ce + dice_loss  # Paper's λ=1 for both terms





# volume_indices = list(range(131))  # All LITS17 volumes
volume_indices = list(range(2)) #test small subset
train_volumes, test_volumes = train_test_split(volume_indices, test_size=0.2, random_state=42)
print(len(train_volumes), len(test_volumes))

#5 minutes for loading data
print("aaa")
train_loader = DataLoader(MedicalSliceDataset_2D("data/LITS17", file_indices = train_volumes), batch_size=4, shuffle=True) 
print(12111111)
test_loader = DataLoader(MedicalSliceDataset_2D("data/LITS17", file_indices = test_volumes), batch_size=4, shuffle=True) 
print(1211111)






model = UNet2D(n_channels=1, n_classes=3)
criterion = hybrid_loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)


#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_dice = 0

for epoch in range(500):
    print(f"\nEpoch {epoch+1}/500")
    model.train()
    train_losses = []

    # Training with tqdm progress bar
    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = hybrid_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print(f"  Train loss: {sum(train_losses)/len(train_losses):.4f}")

    # Validation with tqdm progress bar
    model.eval()
    class_dice = [0.0, 0.0, 0.0]
    with torch.no_grad():
        with tqdm(test_loader, desc="Validation", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device).squeeze(1)
                outputs = model(inputs)
                batch_scores = dice_score_per_class(outputs, targets)
                for i in range(3):
                    class_dice[i] += batch_scores[i]
    
    avg_dice = [c/len(test_loader) for c in class_dice]
    mean_dice = sum(avg_dice) / len(avg_dice)
    print(f"  Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}, Mean: {mean_dice:.4f}")

    # Save best model based on tumor Dice (class 2)
    if avg_dice[2] > best_dice:
        best_dice = avg_dice[2]
        torch.save(model.state_dict(), "best_model.pth")

print(f"\nBest Tumor Dice: {best_dice*100:.2f}%")