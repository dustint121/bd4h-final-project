import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from classes.nnUNet_class import nnUNet3D
from classes.VolumeDataset_3D import MedicalVolumeDataset_3D
from utils_3D_models import hybrid_loss, dice_score_3d, dice_score_3d_perclass


# --- Training and Validation Loop ---
# Training setup
volume_indices = list(range(2))  # Test with small subset
train_volumes, test_volumes = train_test_split(volume_indices, test_size=0.2, random_state=42)

train_dataset = MedicalVolumeDataset_3D("data/LITS17", train_volumes)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = MedicalVolumeDataset_3D("data/LITS17", test_volumes)
test_loader = DataLoader(test_dataset, batch_size=2)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nnUNet3D(n_channels=1, n_classes=3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)



best_dice = 0

for epoch in range(500):
    print(f"\nEpoch {epoch+1}/500")
    model.train()
    train_losses = []

    # Training with tqdm
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

    print(f" Train loss: {sum(train_losses)/len(train_losses):.4f}")

    # Validation with tqdm and per-class Dice
    model.eval()
    class_dice = [0.0, 0.0, 0.0]
    with torch.no_grad():
        with tqdm(test_loader, desc="Validation", leave=False) as pbar:
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device).squeeze(1)
                outputs = model(inputs)
                batch_scores = dice_score_3d_perclass(outputs, targets, num_classes=3)
                for i in range(3):
                    class_dice[i] += batch_scores[i]

    avg_dice = [c/len(test_loader) for c in class_dice]
    mean_dice = sum(avg_dice) / len(avg_dice)
    print(f" Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}, Mean: {mean_dice:.4f}")

    # Save best model based on tumor Dice (class 2)
    if avg_dice[2] > best_dice:
        best_dice = avg_dice[2]
        torch.save(model.state_dict(), "best_model.pth")
        print(f"\nBest Tumor Dice: {best_dice*100:.2f}%")