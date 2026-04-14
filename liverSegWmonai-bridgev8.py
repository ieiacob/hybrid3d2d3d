#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Emil
"""
#%%----------------- Imports ----------------
import os
import re
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Orientationd, Spacingd, ScaleIntensityRanged, SaveImage,
    SpatialCrop
)
from monai.transforms.utils import generate_spatial_bounding_box
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import MapTransform, ResizeWithPadOrCrop

#%% ==========================================
# 1. The Failsafe (Must be included here too)
# ==========================================
class SyncFailsafed(MapTransform):
    def __init__(self, keys, image_key="image", label_key="label"):
        super().__init__(keys)
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        if hasattr(d[self.label_key], 'affine') and hasattr(d[self.image_key], 'affine'):
            d[self.label_key].affine = d[self.image_key].affine.clone()
            
        img_shape = d[self.image_key].shape[1:] 
        lbl_shape = d[self.label_key].shape[1:]
        
        if img_shape != lbl_shape:
            matcher = ResizeWithPadOrCrop(spatial_size=img_shape)
            d[self.label_key] = matcher(d[self.label_key])
            
        return d

#%% ==========================================
# 2. Configuration & Pairing: adjust according to local configuration
# ==========================================
IMAGE_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/volumes/'
MASK_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/masks/segmentations/'

data_dir = "/media/volume/Data/Projects/LITS/data/LiverTumor/volumes/"
output_dir = "/media/volume/Data/Projects/LITS/data/LiverTumor/LiverOnlyDataset" # <-- NEW FOLDER FOR CROPPED DATA

os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

def extract_number(filepath):
    match = re.search(r'\d+', os.path.basename(filepath))
    return int(match.group()) if match else -1

train_images = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])
train_labels = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])


image_dict = {extract_number(img): img for img in train_images}
label_dict = {extract_number(lbl): lbl for lbl in train_labels}

data_dicts = []
for patient_id, img_path in image_dict.items():
    if patient_id in label_dict:
        data_dicts.append({
            "id": patient_id,
            "image": img_path,
            "label": label_dict[patient_id]
        })

print(f"Found {len(data_dicts)} patients to process.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPS = 96 # Must match what the Scout was trained on

#%% ==========================================
# 3. Transforms
# ==========================================
pre_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    SyncFailsafed(keys=["label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
])

# MONAI's saver utility (One for images, one for labels)
img_saver = SaveImage(output_dir=os.path.join(output_dir, "imagesTr"), output_ext=".nii.gz", output_postfix="", separate_folder=False, resample=False)
lbl_saver = SaveImage(output_dir=os.path.join(output_dir, "labelsTr"), output_ext=".nii.gz", output_postfix="", separate_folder=False, resample=False)
#%% ==========================================
# 4. Load the Trained Scout
# ==========================================
model = SegResNet(
    spatial_dims=3, 
    init_filters=16, 
    in_channels=1, 
    out_channels=1, 
    dropout_prob=0.2
).to(device)

# Load the weights once Phase 1 finishes
model.load_state_dict(torch.load("models/best_scout_model.pth"))
model.eval()

#%% ==========================================
# 5. The Cropping Engine
# ==========================================
margin = [10, 10, 10] # Add a 10-pixel buffer around the liver, just in case

with torch.no_grad():
    for data in data_dicts:
        patient_id = data["id"]
        print(f"Processing Patient {patient_id}...")
        
        # Run the preprocessing pipeline
        processed_data = pre_transforms(data)
        img_tensor = processed_data["image"].unsqueeze(0).to(device) # Add batch dim
        
        # Predict the Liver Mask
        with torch.cuda.amp.autocast():
            pred_logits = sliding_window_inference(img_tensor, (SPS, SPS, SPS), 4, model)
            
        pred_mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze(0).cpu() # Remove batch dim
        
        # Find the bounding box of the predicted liver
        try:
            box_start, box_end = generate_spatial_bounding_box(pred_mask, margin=margin)
        except ValueError:
            print(f"?? Scout failed to find a liver in Patient {patient_id}. Skipping crop.")
            continue
            
        # Crop the Image and the Labels safely
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        
        cropped_img = cropper(processed_data["image"])
        cropped_lbl = cropper(processed_data["label"])
        
        # Save them to the new folder
        img_meta = cropped_img.meta
        lbl_meta = cropped_lbl.meta
        
        # Inject the desired filename directly into the metadata dictionary
        img_meta["filename_or_obj"] = f"liver_{patient_id}.nii.gz"
        lbl_meta["filename_or_obj"] = f"liver_{patient_id}.nii.gz"
        
        # Save using the dedicated savers
        img_saver(cropped_img, meta_data=img_meta)
        lbl_saver(cropped_lbl, meta_data=lbl_meta)
        
print("Bridging/Transforming complete! Ready for the last Phase.")
