#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascaded Pipeline approach

@author: Emil
"""
#%%------------------- Imports ------------------
import os
#import glob
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged,
    SpatialPadd, RandCropByPosNegLabeld, RandFlipd, RandRotated, Lambdad,
    MapTransform, ResizeWithPadOrCrop,
    EnsureTyped
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

#%% ==========================================
# 1. Configuration: must be adjusted as appropriate
# ==========================================
IMAGE_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/volumes/'
MASK_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/masks/segmentations/'

#data_dir = "/path/to/your/lits/data" # <-- UPDATE THIS
#train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
#data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

images = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])
labels = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]


# Simple 80/20 split for validation during training
train_files, val_files = data_dicts[:-20], data_dicts[-20:]

SPS = 96 # Spatial Size for the scout model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% ==========================================
# 2. Transforms (The Scout Setup)
# ==========================================
class SyncFailsafed(MapTransform):
    """
    Forces the label to perfectly match the image's metadata AND spatial dimensions.
    """
    def __init__(self, keys, image_key="image", label_key="label"):
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        
        # Force the label to inherit the image's exact physical coordinates
        if hasattr(d[self.label_key], 'affine') and hasattr(d[self.image_key], 'affine'):
            d[self.label_key].affine = d[self.image_key].affine.clone()
            
        # The Shape Override (adapts if missing slices)
        img_shape = d[self.image_key].shape[1:] 
        lbl_shape = d[self.label_key].shape[1:]
        
        if img_shape != lbl_shape:
            matcher = ResizeWithPadOrCrop(spatial_size=img_shape)
            d[self.label_key] = matcher(d[self.label_key])
            
        return d
    
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),

    # Fixes masks that are missing slices, if any
    SyncFailsafed(keys=["label"]),
    
    # Converts 1 (Liver) and 2 (Tumor) into a single 1.0 (Foreground)
    Lambdad(keys=["label"], func=lambda x: (x > 0).float()),
    
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    
    # padding
    SpatialPadd(keys=["image", "label"], spatial_size=(SPS, SPS, SPS)),
    
    # cripping
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(SPS, SPS, SPS), 
        pos=1, neg=1, num_samples=4,
        allow_smaller=False,
        image_key="image", image_threshold=0
    ),
    
    RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
    RandRotated(keys=["image", "label"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.3, mode=("bilinear", "nearest"))
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),

    # Fixes masks that are missing slices, if any
    SyncFailsafed(keys=["label"]),
    
    # the scout merger
    Lambdad(keys=["label"], func=lambda x: (x > 0).float()),
    
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
])

#%% ==========================================
# 3. DataLoaders
# ==========================================
train_ds = CacheDataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)

val_ds = CacheDataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=8) #adjust workers, as available

#%% ==========================================
# 4. Model, Loss, Optimizer
# ==========================================
# out_channels=1 because we are predicting a single binary mask
model = SegResNet(
    spatial_dims=3, 
    init_filters=16, 
    in_channels=1, 
    out_channels=1, 
    dropout_prob=0.2
).to(device)

loss_function = DiceLoss(sigmoid=True) # Sigmoid for binary classification
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")
scaler = torch.cuda.amp.GradScaler()

#%% ==========================================
# 5. The Training Loop
# ==========================================
max_epochs = 150
best_metric = -1

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        
    print(f"Train Loss: {epoch_loss/step:.4f}")
    
    # Validation 
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (SPS, SPS, SPS), 4, model)
                
                # Convert output logits to binary mask (0 or 1)
                val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            print(f"Validation Dice: {metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), "models/best_scout_model.pth")
                print("Saved new best scout model!")
