from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from classes.UNET2_5D_class import MedicalVolumeDataset2_5D, UNet2_5D


def hybrid_loss(pred, target):
    ce = F.cross_entropy(pred, target)
    pred_prob = F.softmax(pred, dim=1)
    dice = 1 - dice_score_2d(pred_prob, target)
    return ce + dice

def dice_score_2d(pred, target):
    smooth = 1e-6
    pred = pred.argmax(1)
    scores = []
    for class_idx in range(3):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        scores.append((2.*intersection + smooth)/(union + smooth))
    return scores  # Return list of scores instead of mean
    # return torch.mean(torch.tensor(scores))

# Training setup
volume_indices = list(range(2))  # Test with small subset
train_volumes, test_volumes = train_test_split(volume_indices, test_size=0.2, random_state=42)

train_dataset = MedicalVolumeDataset2_5D("data/LITS17", train_volumes, num_slices=3)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = MedicalVolumeDataset2_5D("data/LITS17", test_volumes, num_slices=3)
test_loader = DataLoader(test_dataset, batch_size=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2_5D(in_channels=3, n_classes=3).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

# Training loop (similar to 3D version but with 2D processing)
from tqdm import tqdm

for epoch in range(500):
    model.train()
    epoch_loss = 0.0
    
    # Training with progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/500 [Train]", leave=False) as pbar:
        for batch_idx, (data, targets) in enumerate(pbar):
            # Flatten slices into batch dimension
            batch_size, num_slices, channels, height, width = data.shape
            data = data.view(-1, channels, height, width).to(device)
            targets = targets.view(-1, height, width).to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = hybrid_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Validation with progress bar
    model.eval()
    with torch.no_grad():
        class_dice = [0.0, 0.0, 0.0]  # [background, organ, tumor]
        with tqdm(test_loader, desc=f"Epoch {epoch+1}/500 [Val]", leave=False) as pbar:
            for data, targets in pbar:
                data = data.view(-1, 3, height, width).to(device)
                targets = targets.view(-1, height, width).to(device)
                outputs = model(data)
                
                batch_scores = dice_score_2d(outputs, targets)
                for i in range(3):
                    class_dice[i] += batch_scores[i]
                
                pbar.set_postfix({
                    "bg": f"{batch_scores[0]:.2f}",
                    "organ": f"{batch_scores[1]:.2f}",
                    "tumor": f"{batch_scores[2]:.2f}"
                })
        
        # Calculate averages
        avg_dice = [c/len(test_loader) for c in class_dice]
        mean_dice = sum(avg_dice) / len(avg_dice)
        print(f"Epoch {epoch+1} \t Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"  Dice - Background: {avg_dice[0]:.4f}, Organ: {avg_dice[1]:.4f}, Tumor: {avg_dice[2]:.4f}, Mean: {mean_dice:.4f}")