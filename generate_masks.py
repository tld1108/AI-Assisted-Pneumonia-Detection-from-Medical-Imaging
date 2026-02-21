import cv2
import os
import numpy as np

def create_pneumonia_mask(img_path, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    
    _, mask = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    cv2.imwrite(output_path, mask.astype(np.uint8))

for class_name in ['NORMAL', 'PNEUMONIA']:
    img_dir = f'data/chest_xray/train/{class_name}'
    mask_dir = f'masks/train/{class_name}'
    
    images = os.listdir(img_dir)[:500]  
    print(f"Generating {len(images)} masks for {class_name}...")
    
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        create_pneumonia_mask(img_path, mask_path)

print("all mask generated..")
