#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Emil
"""
#%%------------- imports ------------
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from monai.transforms import LoadImage, EnsureChannelFirst, Resize
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
#%%--------------- Functions --------------
def evaluate_test_predictions(test_mask_dir, pred_dir, output_dir, num_classes=3):
    """
    Reads saved NIfTI predictions and ground truths, computes per-case numerical 
    metrics (Dice and HD95), and generates statistical visualizations.
    """
    print("\n==========================================")
    print("      STARTING STATISTICAL EVALUATION     ")
    print("==========================================")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MONAI metrics
    # reduction="none" ensures we get scores for each individual image, not just a batch average
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    
# Tools to load NIfTI files directly
    loader = LoadImage(image_only=True)
    channel_first = EnsureChannelFirst()
    
    # ADD THIS: Force everything to the same shape.
    # We must use "nearest" mode so we don't accidentally blend our discrete 0, 1, 2 classes!
    #resizer = Resize(spatial_size=(256, 256, 64), mode="nearest")
    
    # Get paired files
    true_masks = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(('.nii', '.nii.gz'))])
    
    results = []

    for mask_file in true_masks:
        true_path = os.path.join(test_mask_dir, mask_file)
        
        # Extract the ID number from the mask filename (e.g., '0' from 'segmentation-0.nii')
        match = re.search(r'\d+', mask_file)
        if not match:
            print(f"Skipping {mask_file}: Could not find an ID number in the filename.")
            continue
            
        case_id = match.group()
        
        # Construct the exact expected MONAI prediction filename
        expected_pred_name = f"volume-{case_id}_pred.nii.gz"
        pred_path = os.path.join(pred_dir, expected_pred_name)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Looked for {expected_pred_name} to match {mask_file}, but couldn't find it. Skipping.")
            continue
        
        # Load data
        true_tensor = loader(true_path)
        pred_tensor = loader(pred_path)
        
        # FIX 1: Strip away any rogue dimensions and force [H, W, D]
        true_tensor = true_tensor.squeeze()
        pred_tensor = pred_tensor.squeeze()
        
        # Manually add the single channel dimension -> [1, H, W, D]
        true_tensor = true_tensor.unsqueeze(0)
        pred_tensor = pred_tensor.unsqueeze(0)
        
        # FIX 2: Dynamic Spatial Resizing
        # Grab the exact [H, W, D] native shape of the patient's ground truth mask
        target_spatial_shape = true_tensor.shape[1:]
        
        # Create a dynamic resizer for this specific patient
        dynamic_resizer = Resize(spatial_size=target_spatial_shape, mode="nearest")
        
        # Resize ONLY the prediction to match the ground truth
        pred_tensor = dynamic_resizer(pred_tensor)
        
        # Add batch dimension: [1, H, W, D] -> [1, 1, H, W, D]
        true_tensor = true_tensor.unsqueeze(0)
        pred_tensor = pred_tensor.unsqueeze(0)
        
        # One-hot encode for metrics: [1, C, H, W, D]
        true_onehot = one_hot(true_tensor, num_classes=num_classes)
        pred_onehot = one_hot(pred_tensor, num_classes=num_classes)
        
        # Compute metrics (returns tensor of shape [Batch, Classes])
        dice_score = dice_metric(y_pred=pred_onehot, y=true_onehot)[0]
        hd95_score = hd95_metric(y_pred=pred_onehot, y=true_onehot)[0]
        
        # Store results (Index 0 is Liver, Index 1 is Tumor since background is excluded)
        case_id = mask_file.replace('.nii.gz', '').replace('.nii', '')
        
        # Handle cases where a tumor might not exist in the ground truth
        tumor_dice = dice_score[1].item()
        tumor_hd95 = hd95_score[1].item()
        
        results.append({
            "Case_ID": case_id,
            "Liver_Dice": dice_score[0].item(),
            "Tumor_Dice": tumor_dice,
            "Liver_HD95": hd95_score[0].item(),
            "Tumor_HD95": tumor_hd95 if not torch.isinf(hd95_score[1]) else None
        })
        print(f"Evaluated: {case_id}")
        
    # --- 1. NUMERICAL RESULTS (CSV) ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "test_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n--- Summary Statistics ---")
    print(df.describe().drop("count")) # Prints mean, std, min, max, etc.
    
    # --- 2. VISUALIZATIONS ---
    sns.set_theme(style="whitegrid")
    
    # Melt dataframe for easier Seaborn plotting
    df_dice = df.melt(id_vars=["Case_ID"], value_vars=["Liver_Dice", "Tumor_Dice"], 
                      var_name="Category", value_name="Dice Score")
    df_hd95 = df.melt(id_vars=["Case_ID"], value_vars=["Liver_HD95", "Tumor_HD95"], 
                      var_name="Category", value_name="HD95 (mm)")

    # Plot 1: Dice Score Distributions
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_dice, x="Category", y="Dice Score", palette="Set2", showmeans=True, 
                meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    sns.stripplot(data=df_dice, x="Category", y="Dice Score", color=".25", alpha=0.6, jitter=True)
    plt.title("Dice Score Distribution across Test Set", fontsize=16)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(output_dir, "dice_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Hausdorff Distance Distributions
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_hd95.dropna(), x="Category", y="HD95 (mm)", palette="Set1", inner="quartile")
    sns.stripplot(data=df_hd95.dropna(), x="Category", y="HD95 (mm)", color=".25", alpha=0.6, jitter=True)
    plt.title("95th Percentile Hausdorff Distance (HD95) across Test Set\n(Lower is better)", fontsize=16)
    plt.savefig(os.path.join(output_dir, "hd95_violinplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n==========================================")
    print(f"Results saved to: {output_dir}")
    print(f"- test_metrics_summary.csv")
    print(f"- dice_boxplot.png")
    print(f"- hd95_violinplot.png")
    print(f"==========================================")

#%% ==========================================
# Run it:
# ==========================================
# Ensure these paths match where your previous script saved its outputs
TEST_MASK_DIR = '/media/volume/Data/Projects/LITS/data/LiverTumor/test/masks/'   
PREDICTION_DIR = './final_test_predictions4/' 
OUTPUT_METRICS_DIR = './final_test_predictions4/quantitative/' 


evaluate_test_predictions(
    test_mask_dir=TEST_MASK_DIR,
    pred_dir=PREDICTION_DIR,
    output_dir=OUTPUT_METRICS_DIR
)
