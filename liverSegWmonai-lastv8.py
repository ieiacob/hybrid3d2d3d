#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascaded Pipeline approach

@author: Emil
"""
#%%------------------ Imports - ------
import os
import glob
import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    RandRotated,
    RandFlipd, AsDiscrete, SpatialPadd,
    RandCropByLabelClassesd,
)
from monai.networks.nets import SegResNet
#from monai.losses import DiceCELoss
#from monai.losses import DiceFocalLoss
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter

#%% ==========================================
# 1. Configuration & New Directories
# ==========================================
# UPDATE THIS to LiverOnlyDataset folder, as needed
data_dir = "/media/volume/Data/Projects/LITS/data/LiverTumor/LiverOnlyDataset" 

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

# Create data dictionaries
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

# Simple 80/20 train/val split
split = int(len(data_dicts) * 0.8)
train_files, val_files = data_dicts[:split], data_dicts[split:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPS = 96 # Crop size

print(f"Training on {len(train_files)} files, Validating on {len(val_files)} files.")

#%% ==========================================
# 2. Transforms (No Spacing/Orientation)
# ==========================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    
    SpatialPadd(keys=["image", "label"], spatial_size=(SPS, SPS, SPS), mode="constant", value=0),
    
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(SPS, SPS, SPS),
        num_classes=3,
        # The 'ratios' parameter tells MONAI what to center the crops on.
        # [1, 2, 3] means out of 6 samples:
        # - 1 patch centered on Background (Class 0)
        # - 2 patches centered on Liver (Class 1)
        # - 3 patches centered on Tumor (Class 2)
        ratios=[1, 1, 2],
        num_samples=6,
    ),

    RandRotated(keys=["image", "label"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, keep_size=True),
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
])


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

#%% ==========================================
# 3. Data Loaders
# ==========================================
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=16)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=16)

#%% ==========================================
# 4. Multi-Class Model, Loss, and Metrics
# ==========================================
class CustomDiceCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        # Dice part: We still want to ignore background for the Dice calculation
        self.dice = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        # Cross-Entropy part: apply heavy penalties here
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        
        # PyTorch CrossEntropy expects target shape (Batch, H, W, D) 
        # MONAI loads targets as (Batch, 1, H, W, D). Squeezing removes that extra '1' dimension.
        targets_ce = targets.squeeze(1).long()
        ce_loss = self.ce(inputs, targets_ce)
        
        return dice_loss + ce_loss

# out_channels=3 for Background (0), Liver (1), Tumor (2)
model = SegResNet(
    spatial_dims=3, 
    init_filters=16, 
    in_channels=1, 
    out_channels=3, 
    dropout_prob=0.2
).to(device)


# Create class weights: 10% Background, 30% Liver, 60% Tumor
weights = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32).to(device)
## [Background, Liver, Tumor]
weights = torch.tensor([0.02, 0.18, 0.80], dtype=torch.float32).to(device)

loss_function = CustomDiceCELoss(
    class_weights=weights
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

max_epochs = 300

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# Metrics: include_background=False ignores class 0. 
# We track classes 1 and 2.
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Post-processing to convert model outputs to distinct classes for metric calculation
post_pred = Compose([AsDiscrete(argmax=True), AsDiscrete(to_onehot=3)])
post_label = Compose([AsDiscrete(to_onehot=3)])

#%% ==========================================
# 5. The Training Loop
# ==========================================
# Create a TensorBoard writer (for visualizations during and after)
writer = SummaryWriter(log_dir="runs/Train25Dmodel")


val_interval = 10
best_metric = -1
best_metric_epoch = -1

scaler = torch.cuda.amp.GradScaler()

for epoch in range(max_epochs):
    print(f"--- Epoch {epoch + 1}/{max_epochs} ---")
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
            
        # Scale the loss, run backward, step optimizer, and update scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        print(f"   Step {step}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
    epoch_loss /= step
    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

    writer.add_scalar("Loss/Train", epoch_loss, epoch + 1)

    # Validation Step
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, (SPS, SPS, SPS), 4, model)
                
                # Deconstruct the batch and apply post-processing
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                
                # Compute metric
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            # Aggregate the metrics
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            
            print(f"Validation Dice (Average of Liver + Tumor): {metric:.4f}")

            # Log validation score
            writer.add_scalar("Dice/Validation", metric, epoch + 1)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "models/best_tumor_expert4.pth")
                print(f"*** New Best Model Saved at Epoch {best_metric_epoch} ***")

    # End of the epoch (after processing all batches)
    scheduler.step()
    
    # Print the current learning rate to watch it change
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1} completed. Current LR: {current_lr:.6f}")

print(f"Training Complete! Best Metric: {best_metric:.4f} at Epoch: {best_metric_epoch}")
writer.close() # NEW: Close the writer
