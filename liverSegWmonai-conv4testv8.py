#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Emil
"""
#%%-- Imports ---------------
import os
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRanged, AsDiscrete, SaveImage, Spacingd,
    Orientationd, Invertd, SaveImaged,
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms.utils import generate_spatial_bounding_box
from monai.networks.utils import one_hot


#%% 1. Setup (adjust, as appropriate)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPS = 96  # Spatial size from your Phase 2 training

IMAGE_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/test/volumes/'
MASK_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/test/masks/'
OUTPUT_DIR = "./final_test_predictions4"

images = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])
labels = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')])
test_files = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

#%% 2. Pipeline (Clean online scaling, NO cropping transforms here)
# --------------------------------------------------
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    # The models must see RAS orientation during inference
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    # The models must see this exact spacing during inference
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    # Phase 2 expects 0.0 to 1.0 windowed intensities
    ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=250, b_min=0.0, b_max=1.0, clip=True),
])

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=8)

#%% 3. Load BOTH Models
# --------------------------------------------------
# Must match Phase 1 model architecture
phase1_model = SegResNet(
    spatial_dims=3, 
    init_filters=16, 
    in_channels=1, 
    out_channels=1, 
    dropout_prob=0.2
).to(device)
phase1_model.load_state_dict(torch.load("models/best_scout_model.pth"))
phase1_model.eval()

# Must match Phase 2 Tumor Expert model
phase2_model = SegResNet(
    spatial_dims=3, 
    init_filters=16, 
    in_channels=1, 
    out_channels=3, 
    dropout_prob=0.2
).to(device)
phase2_model.load_state_dict(torch.load("models/best_tumor_expert4.pth"))
phase2_model.eval()

#%% 4. Metrics & Post-Processing
# --------------------------------------------------
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
post_label = Compose([AsDiscrete(to_onehot=3)])

dice_metric = DiceMetric(include_background=False, reduction="mean")

output_dir = OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# MONAI's tool to save the physical NIfTI files
saver = SaveImage(output_dir=output_dir, output_postfix="pred", output_ext=".nii.gz", resample=False, separate_folder=False)

# Create the CSV file for some statistics
csv_path = os.path.join(output_dir, "best_slices_report.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
#csv_writer.writerow(["Patient_ID", "Best_Slice_Z", "Tumor_Dice_Score"])
csv_writer.writerow(["Patient_ID", "3D_Liver_Dice", "3D_Tumor_Dice", "Best_Slice_Z", "Best_Slice_Tumor_Dice"])

#%% 5. The Cascaded Evaluation Loop
# --------------------------------------------------
# Create a list to store our slice data
best_slice_records = []
volume_records = []

print(f"Starting cascaded inference on {len(test_files)} patients...")

with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        # Raw, uncropped inputs and the hidden truth labels
        inputs, true_labels = test_data["image"].to(device), test_data["label"].to(device)
        
        with torch.cuda.amp.autocast():
            # ========================================================
            # The check points can be removed, as appropriate.
            # ========================================================
            print(f"\n Check Point: PATIENT {i+1} ---")
            print(f"1. Input Shape: {inputs.shape}")
            print(f"2. Input Min/Max: {inputs.min().item():.4f} to {inputs.max().item():.4f}")
            if torch.isnan(inputs).any():
                print(">>> WARNING: INPUT CONTAINS NaNs! The transforms broke the image.")

            phase1_outputs = sliding_window_inference(
                inputs=inputs, roi_size=(96, 96, 96), sw_batch_size=4, predictor=phase1_model
            )
            
            print(f"3. Phase 1 Logits Min/Max: {phase1_outputs.min().item():.4f} to {phase1_outputs.max().item():.4f}")
            if torch.isnan(phase1_outputs).any():
                print(">>> WARNING: OUTPUT CONTAINS NaNs! Autocast or model weights failed.")
            
            #phase1_preds = torch.argmax(phase1_outputs, dim=1, keepdim=True)
            # Converts logits to a 0-1 percentage, then keeps anything over 50%
            phase1_preds = (torch.sigmoid(phase1_outputs) > 0.5).int()
            print(f"4. Total Liver Pixels Predicted: {phase1_preds.sum().item()}")            
            # ========================================================
            # Generate the "Legal" Crop Box
            # ========================================================
            # If Phase 1 found a liver, calculate its 3D bounding box
            if phase1_preds.sum() > 0:
                # generate_spatial_bounding_box expects shape [C, H, W, D], so we pass pred[0]
                box_start, box_end = generate_spatial_bounding_box(phase1_preds[0])
                
                # Apply a slight margin (e.g., 10 voxels) so the Phase 2 model has edge context
                margin = 10
                for d in range(3):
                    box_start[d] = max(0, box_start[d] - margin)
                    box_end[d] = min(inputs.shape[d+2], box_end[d] + margin)
                    
                # Crop the input image dynamically based on the Phase 1 prediction
                cropped_inputs = inputs[
                    :, :, 
                    box_start[0]:box_end[0], 
                    box_start[1]:box_end[1], 
                    box_start[2]:box_end[2]
                ]
                
                # Crp the hidden ground truth label perfectly to match, for fair grading later
                cropped_labels = true_labels[
                    :, :, 
                    box_start[0]:box_end[0], 
                    box_start[1]:box_end[1], 
                    box_start[2]:box_end[2]
                ]
                
                print(f" Check Point: Phase 1 found Liver. Crop Box: {box_start} to {box_end}")
            else:
                print(" Check Point: Phase 1 FAILED. No liver found. Passing full image.")                # If Phase 1 completely failed and hallucinated nothing, we skip cropping
                # and pass the whole image (which usually means a low score, but that's reality)
                cropped_inputs = inputs
                cropped_labels = true_labels

            # ========================================================
            # The Fine Pass (Phase 2 Tumor Expert)
            # ========================================================
            phase2_outputs = sliding_window_inference(
                inputs=cropped_inputs, 
                roi_size=(SPS, SPS, SPS), 
                sw_batch_size=4, 
                overlap=0.5,
                mode="gaussian",
                predictor=phase2_model
            )
            phase2_raw_guess = torch.argmax(phase2_outputs, dim=1)
            print(f" Check Point: Unique classes predicted by Phase 2: {torch.unique(phase2_raw_guess).tolist()}")            
            print(f" Check Point: Phase 2 Liver Pixels: {(phase2_raw_guess == 1).sum().item()}")
            print(f" Check Point: Phase 2 Tumor Pixels: {(phase2_raw_guess == 2).sum().item()}")

            # ========================================================
            # Reconstruct and Past
            # ========================================================
            # Create a blank canvas
            final_pred_full_size = torch.zeros((1, 3, inputs.shape[2], inputs.shape[3], inputs.shape[4]), device=device)
            
            # Make the blank canvas highly confident that it is "Background" (Class 0)
            final_pred_full_size[:, 0, ...] = 10.0   # Background channel gets a massive positive logit
            final_pred_full_size[:, 1:, ...] = -10.0 # Liver and Tumor channels get massive negative logits

            # Restore the CT scanner's physical coordinates!
            from monai.data import MetaTensor
            if isinstance(inputs, MetaTensor):
                final_pred_full_size = MetaTensor(final_pred_full_size, affine=inputs.affine, meta=inputs.meta)

            # Paste the Phase 2 predictions back into the correctly aligned physical space
            if phase1_preds.sum() > 0:
                final_pred_full_size[
                    :, :, 
                    box_start[0]:box_end[0], 
                    box_start[1]:box_end[1], 
                    box_start[2]:box_end[2]
                ] = phase2_outputs
            else:
                final_pred_full_size = phase2_outputs        


        # ========================================================
        # One-Hot Encoding
        # ========================================================
        
        # Convert Phase 2's logits into discrete class indices (0, 1, or 2)
        # Shape changes from [1, 3, H, W, D] -> [1, 1, H, W, D]
        final_classes = torch.argmax(final_pred_full_size, dim=1, keepdim=True)
        
        # Expand Predictions into One-Hot format
        # Shape changes to [1, 3, H, W, D] where channels are strictly 0s and 1s
        pred_onehot = one_hot(final_classes, num_classes=3)
        
        # Expand Ground Truth Labels into One-Hot format
        # (Assuming your ground truth variable is named `true_labels` and shaped [1, 1, H, W, D])
        label_onehot = one_hot(true_labels.long(), num_classes=3)
        
        # Calculate Multi-Class Dice (ignoring the Background class 0)
        # reduction="mean_batch" keeps the classes separated!
        metric_calculator = DiceMetric(include_background=False, reduction="mean_batch")
        metric_calculator(y_pred=pred_onehot, y=label_onehot)
        
        # Extract the scores!
        scores = metric_calculator.aggregate() 
        liver_dice = scores[0].item()
        tumor_dice = scores[1].item()
        
        print(f" Check Point: Corrected Liver Dice: {liver_dice:.4f}")
        print(f" Check Point: Corrected Tumor Dice: {tumor_dice:.4f}")
        
        metric_calculator.reset()

        # ========================================================
        test_outputs_list = decollate_batch(final_pred_full_size)
        test_labels_list = decollate_batch(true_labels) 
        
        metric_preds = [post_pred(j) for j in test_outputs_list]
        metric_labels = [post_label(j) for j in test_labels_list]
        
        # ========================================================
        # Capture the 3D Dice scores for this specific patient
        # ========================================================
        patient_metrics = dice_metric(y_pred=metric_preds, y=metric_labels)
        
        # patient_metrics is a tensor of shape [1, 2] (Batch size 1, Liver and Tumor)
        liver_dice_3d = patient_metrics[0, 0].item()
        
        # If a patient has NO tumor in the ground truth, MONAI returns NaN (Not a Number)
        if torch.isnan(patient_metrics[0, 1]):
            tumor_dice_3d = "N/A"
        else:
            tumor_dice_3d = patient_metrics[0, 1].item()
            
        # Save the full 3D NIfTI file
        # ========================================================
        
        # Unpack the batched dataloader dictionary into a list of single patient dictionaries
        test_data_decollated = decollate_batch(test_data)

        for i, pred in enumerate(test_outputs_list):
            
            # Grab this specific patient's unbatched dictionary
            patient_dict = test_data_decollated[i]
            
            # Get the discrete class predictions (0, 1, 2)
            # pred is shape [C, H, W, D], argmax makes it [1, H, W, D]
            pred_label = torch.argmax(pred, dim=0, keepdim=True)
            
            # Add the prediction into the single patient's dictionary 
            patient_dict["pred"] = pred_label
            
            # Create the Inverter
            inverter = Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=True, 
                to_tensor=True,
            )
            
            # un-warp the prediction back to the original CT space
            inverted_patient = inverter(patient_dict)
            
            # Save the un-warped prediction safely to disk
            saver = SaveImaged(
                keys="pred", 
                output_dir="./final_test_predictions4/", 
                output_postfix="pred", 
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False
            )
            
            # Execute the save using the correctly unbatched and inverted data
            saver(inverted_patient)
        # ========================================================
        # Slice Analysis for Best Tumor Dice
        # ========================================================
        # Convert tensors to numpy arrays of shape [X, Y, Z]
        pred_classes = torch.argmax(test_outputs_list[0], dim=0).cpu().numpy() 
        true_classes = test_labels_list[0][0].cpu().numpy() 
        print(f" Check Point: Unique classes in the TRUE label: {np.unique(true_classes)}")

        img_vol = inputs[0, 0].cpu().numpy()
        
        # Extract Patient ID safely from metadata
        patient_path = test_data["image"].meta.get("filename_or_obj", ["Unknown"])[0]
        patient_id = os.path.basename(patient_path).replace(".nii.gz", "")
        
        best_dice = -1.0
        best_z = -1
        
        # Loop through the Z-axis (depth)
        Z_dim = img_vol.shape[2]
        for z in range(Z_dim):
            # Isolate Class 2 (Tumor) into a binary mask
            pred_slice = (pred_classes[:, :, z] == 2).astype(float)
            true_slice = (true_classes[:, :, z] == 2).astype(float)
            
            sum_true = np.sum(true_slice)
            sum_pred = np.sum(pred_slice)
            
            # ONLY grade slices that actually contain ground-truth tumor tissue
            if sum_true > 0:
                intersection = np.sum(pred_slice * true_slice)
                # 2D Dice Formula
                slice_dice = (2.0 * intersection) / (sum_true + sum_pred)
                
                if slice_dice > best_dice:
                    best_dice = slice_dice
                    best_z = z
                    
        # If a tumor was found in this patient, save the results
        if best_z != -1:
            #csv_writer.writerow([patient_id, best_z, f"{best_dice:.4f}"])
            csv_writer.writerow([patient_id, f"{liver_dice_3d:.4f}", f"{tumor_dice_3d:.4f}" if isinstance(tumor_dice_3d, float) else tumor_dice_3d, best_z, f"{best_dice:.4f}"])
            # Append the data to our tracker
            #best_slice_records.append([patient_id, best_z, round(best_dice, 4)])
            best_slice_records.append([patient_id, round(liver_dice_3d, 4), round(tumor_dice_3d, 4) if isinstance(tumor_dice_3d, float) else tumor_dice_3d, best_z, round(best_dice, 4)])
            
            # Create a 3-panel side-by-side image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # .T (Transpose) and origin="lower" orients standard NIfTI arrays upright for viewing
            axes[0].imshow(img_vol[:, :, best_z].T, cmap="gray", origin="lower")
            axes[0].set_title(f"Original CT {patient_id} (Slice {best_z})")
            axes[0].axis("off")
            
            axes[1].imshow((true_classes[:, :, best_z] == 2).T, cmap="gray", origin="lower")
            axes[1].set_title("True Tumor")
            axes[1].axis("off")
            
            axes[2].imshow((pred_classes[:, :, best_z] == 2).T, cmap="gray", origin="lower")
            axes[2].set_title(f"Predicted Tumor (Dice: {best_dice:.4f})")
            axes[2].axis("off")
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{patient_id}_best_slice_{best_z}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            # Handle edge case where the patient has no tumor in the ground truth
            #csv_writer.writerow([patient_id, "No Tumor", "N/A"])
            csv_writer.writerow([patient_id, f"{liver_dice_3d:.4f}", "N/A", "No Tumor", "N/A"])
            best_slice_records.append([patient_id, round(liver_dice_3d, 4), 0.0, -1, 0.0])

        #print(f"Processed Patient {i + 1}/{len(test_loader)} - Best Slice Z: {best_z}")
        if isinstance(tumor_dice_3d, float):
            print(f"Processed Patient {i + 1}/{len(test_loader)}: {patient_id} | 3D Liver: {liver_dice_3d:.4f} | 3D Tumor: {tumor_dice_3d:.4f} | Best Z: {best_z}")
        else:
            print(f"Processed Patient {i + 1}/{len(test_loader)}: {patient_id} | 3D Liver: {liver_dice_3d:.4f} | 3D Tumor: N/A (No True Tumor)")
# 6. Final Honest Score
# --------------------------------------------------
final_metric = dice_metric.aggregate().item()
print("-" * 40)
print(f"*** Final End-to-End Test Set Dice Score: {final_metric:.4f} ***")
csv_file.close()
#%% ==========================================
# Plot the Slice Metrics
# ==========================================

if len(best_slice_records) > 1:
    # Extract data, skipping the header row at index 0
    file_names = [row[0] for row in best_slice_records[1:]]
    dice_scores = [row[2] for row in best_slice_records[1:]]
    
    # Calculate the average slice Dice score
    mean_slice_dice = sum(dice_scores) / len(dice_scores)
    
    # Set up a wide, clean canvas
    plt.figure("Dice Chart", figsize=(12, 6))
    
    # Create the bar chart
    bars = plt.bar(file_names, dice_scores, color="#4C72B0", edgecolor="black", alpha=0.85)
    
    # Add a horizontal dashed line for the average
    plt.axhline(mean_slice_dice, color="#C44E52", linestyle="--", linewidth=2, 
                label=f"Mean Slice Dice: {mean_slice_dice:.4f}")
    
    # Add exact numerical labels floating on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.2f}", 
                 ha="center", va="bottom", fontsize=9)
    
    # Formatting to make it look highly professional
    plt.title("3D Dice Scores per Subject", fontsize=14, fontweight="bold")
    plt.xlabel("Test File Name", fontsize=12)
    plt.ylabel("Dice Score (0.0 to 1.0)", fontsize=12)
    plt.ylim(0, 1.1)  
    plt.xticks(rotation=45, ha="right") 
    plt.grid(axis="y", linestyle="--", alpha=0.7) 
    plt.legend()
    plt.tight_layout() 
    
    # Save the plot alongside the CSV
    plot_path = os.path.join(output_dir, "dice_score_summary_plot.png")
    #plt.show()
    plt.savefig(plot_path, dpi=300) # High resolution
    plt.close()

    print(f"Summary bar chart saved to: {plot_path}")

