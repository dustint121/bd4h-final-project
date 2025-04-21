from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from classes.UNet3D_class import UNet3D
from classes.VolumeDataset_3D import VolumeDataset_3D
from utils_3D_models import hybrid_loss, dice_score_3d



volume_indices = list(range(131))  # All LITS17 volumes
# volume_indices = list(range(5)) #test small subset
train_volumes, test_volumes = train_test_split(volume_indices, test_size=0.5, random_state=42)
# print(len(train_volumes), len(test_volumes))


# Data Loaders
train_dataset = VolumeDataset_3D(dataset_path="data/LITS17", file_indices=train_volumes, depth_fraction=0.1)
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=2,  # Matches paper's LiTS batch size
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # Matches paper's LiTS batch size
    shuffle=True
)

test_dataset = VolumeDataset_3D(dataset_path="data/LITS17", file_indices=test_volumes, depth_fraction=0.1)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=2,  # Matches paper's LiTS batch size
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )
test_loader = DataLoader(
    test_dataset,
    batch_size=2,  # Matches paper's LiTS batch size
    shuffle=True
)


model = UNet3D(n_channels=1, n_classes=3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

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
                batch_scores = dice_score_3d(outputs, targets)
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