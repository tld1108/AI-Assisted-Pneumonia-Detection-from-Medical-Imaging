import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import PneumoniaDataset, train_transform
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataset = PneumoniaDataset(
    'data/chest_xray/train/PNEUMONIA', 
    'masks/train/PNEUMONIA', 
    train_transform
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print(f"train samples: {len(train_dataset)}")

model = smp.Unet(
    encoder_name='resnet34', 
    encoder_weights='imagenet',
    in_channels=1, 
    classes=1
).to(device)

criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("start training...")

for epoch in range(10):  
    model.train()
    total_loss = 0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/10, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    print(f'✅ Epoch {epoch+1} finished! Average Loss: {avg_loss:.4f}')
    
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

print("model saved successfully")