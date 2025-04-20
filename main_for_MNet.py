from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from tqdm import tqdm
from classes.MNet_class import MNet3D
from classes.VolumeDataset_3D import MedicalVolumeDataset_3D
from utils_3D_models import hybrid_loss, dice_score_3d, dice_score_3d_perclass

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
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, targets in tepoch:
            inputs = inputs.to(device)
            targets = targets.to(device).squeeze(1)
            optimizer.zero_grad()
            main_output, aux_outputs = model(inputs)
            loss = hybrid_loss(main_output, targets)
            for i, aux in enumerate(aux_outputs):
                depth_weight = 0.5 ** (5 - i)
                loss += depth_weight * hybrid_loss(aux, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            tepoch.set_description(f"Epoch {epoch+1} [Train]")
            tepoch.set_postfix(loss=loss.item())
    print(f" Train loss: {sum(train_losses)/len(train_losses):.4f}")

    # Validation
    model.eval()
    class_dice = [0.0, 0.0, 0.0]
    with tqdm(test_loader, unit="batch") as vepoch:
        with torch.no_grad():
            for inputs, targets in vepoch:
                inputs = inputs.to(device)
                targets = targets.to(device).squeeze(1)
                main_output, _ = model(inputs)
                batch_score = dice_score_3d_perclass(main_output, targets)
                for i in range(3):
                    class_dice[i] += batch_score[i]
                vepoch.set_description(f"Epoch {epoch+1} [Val]")
                vepoch.set_postfix(dice_tumor=batch_score[2])
    avg_dice = [c/len(test_loader) for c in class_dice]
    mean_dice = sum(avg_dice) / len(avg_dice)
    print(f" Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}, Mean: {mean_dice:.4f}")
    if avg_dice[2] > best_dice:
        best_dice = avg_dice[2]
        torch.save(model.state_dict(), "best_mnet_model.pth")
        print(f"\nBest Tumor Dice: {best_dice*100:.2f}%")
