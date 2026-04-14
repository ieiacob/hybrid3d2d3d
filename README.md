# hybrid3d2d3d
Repository for cascading 3d2d3d liver image segmentation

# Dataset
- Here is one repository (out of many out there) of the dataset: https://github.com/Auggen21/LITS-Challenge
- We selected 15 CT scans (volumes) and the corresponding 15 marks (segments) and stored them in separate folders ("....LITS/data/LiverTumor/test/volumes" and "....LITS/data/LiverTumor/test/masks", respectively). The IDs of these images and masks are: 0, 4, 25, 45, 46, 47, 66, 83, 94, 105, 108, 117, 119, 127, and 130.
- The rest of the images and masks were placed in training folders ("....LITS/data/LiverTumor/volumes" and "....LITS/data/LiverTumor/masks/segmentations", respectively).

# How to train
1. Before running a script all source folders (images, masks, output) must be edited according to the local configuration.
2. Run **liverSegWmonaiv8.py**: saves the best scout model in "./models/best_scout_model.pth" (this is already provided in this repository)
3. Run **liverSegWmonai-bridgev8.py**: uses the model create in the previous run and performs the first stage of image transformations (liver identification). The "liver only" test images and masks are produced for stage 2 (in "....LITS/data/LiverTumor/LiverOnlyDataset/imagesTr" and "....LITS/data/LiverTumor/LiverOnlyDataset/labelsTr" folders, respectively).
4. Run **liverSegWmonai-lastv8.py**: uses "liver only" images for training the "./models/best_tumor_expert4.pth" (provided in this repository).

# How to test, some statistics
1. Run **liverSegWmonai-conv4testv8.py**: it loads both *best_scout_model.pth* and *best_tumor_expert4.pth* models produced during training, uses the 15 "volumes" data saved for testing. It outputs in *./final_test_predictions4* the 15 predicted volumes, which can subsequently be tested agains their respective 15 segments. It also produces some info/statistics in *./final_test_predictions4*.
2. Run **liverSegWmonai-test-statv8.py** to produce some test statistics.
