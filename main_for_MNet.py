from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
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

# Model setup
model = MNet3D(n_channels=1, n_classes=3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
best_tumor_dice = 0
for epoch in range(500):
    print(f"\nEpoch {epoch+1}/500")
    
    # Training phase
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1} [Train]") as train_bar:
        for inputs, targets in train_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).squeeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward
            with autocast():
                main_output, aux_outputs = model(inputs)
                loss = hybrid_loss(main_output, targets)
                
                # Deep supervision with decreasing weights
                for i, aux in enumerate(aux_outputs):
                    loss += (0.5 ** (5-i)) * hybrid_loss(aux, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
    
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Train loss: {avg_train_loss:.4f}")
    
    # Validation phase
    model.eval()
    dice_scores = torch.zeros(3, device=device)
    with tqdm(test_loader, unit="batch", desc=f"Epoch {epoch+1} [Val]") as val_bar:
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).squeeze(1)
                
                # Mixed precision inference
                with autocast():
                    main_output, _ = model(inputs)
                
                # Calculate metrics
                batch_scores = dice_score_3d_perclass(main_output, targets)
                dice_scores += torch.tensor(batch_scores, device=device)
                val_bar.set_postfix(tumor_dice=f"{batch_scores[2]:.4f}")
    
    # Update learning rate
    avg_dice = (dice_scores / len(test_loader)).cpu().numpy()
    scheduler.step(avg_dice[2])  # Monitor tumor dice
    
    print(f"Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}")
    
    # Save best model
    if avg_dice[2] > best_tumor_dice:
        best_tumor_dice = avg_dice[2]
        torch.save(model.state_dict(), "best_mnet_model.pth")
        print(f"New Best Tumor Dice: {best_tumor_dice*100:.2f}%")

    # Early stopping check (optional)
    if optimizer.param_groups[0]['lr'] < 1e-5:
        print("Learning rate too low - stopping training")
        break