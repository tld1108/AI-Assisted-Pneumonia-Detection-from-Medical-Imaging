from dataset import PneumoniaDataset, train_transform
from torch.utils.data import DataLoader

dataset = PneumoniaDataset(
    'data/chest_xray/train/PNEUMONIA', 
    'masks/train/PNEUMONIA', 
    train_transform
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

images, masks = next(iter(dataloader))
print(f"Shape: {images.shape}, {masks.shape}")
print("lanjut training")
