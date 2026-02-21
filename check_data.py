import os
print("🔍 Cek dataset...")
train_normal = len(os.listdir('data/chest_xray/train/NORMAL'))
train_pneumonia = len(os.listdir('data/chest_xray/train/PNEUMONIA'))
print(f"RAIN NORMAL: {train_normal} images")
print(f"TRAIN PNEUMONIA: {train_pneumonia} images")
print("dataset siap......")
