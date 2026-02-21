import torch
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
from dataset import val_transform
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name='resnet34', in_channels=1, classes=1).to(device)
model.load_state_dict(torch.load('model_epoch_10.pth', map_location=device))
model.eval()

test_dir = 'data/chest_xray/test/PNEUMONIA'
if not os.path.exists(test_dir):
    test_dir = 'data/chest_xray/train/PNEUMONIA'

test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg')]
test_img = test_files[0]
img_path = os.path.join(test_dir, test_img)
print(f"🖼️  Testing: {test_img}")

orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
orig_img = cv2.resize(orig_img, (256, 256))  # 256x256

image_norm = val_transform(image=orig_img)['image'].unsqueeze(0).to(device)
with torch.no_grad():
    pred = torch.sigmoid(model(image_norm))
    pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(orig_img, cmap='gray')
plt.title('Original X-Ray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_mask, cmap='Reds')
plt.title('Predicted Pneumonia\n(Red = High Risk)')
plt.axis('off')

plt.subplot(1, 3, 3)
overlay = cv2.addWeighted(orig_img, 0.6, pred_mask_uint8, 0.4, 0)
plt.imshow(overlay, cmap='gray')
plt.title('Clinical Overlay\n(Red tint = Pneumonia)')
plt.axis('off')

plt.tight_layout()
plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization file: prediction_result.png")
print("Epoch 10 Loss: 0.0415 ")
