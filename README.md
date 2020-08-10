# RibFrac-Challenge

Evaluation scripts for Rib Fracture Detection and Classification Challenge (RibFrac).



# Content

```
RibFract-Challenge/
    requirements.txt                Required packages for evaluation
    ribfrac/
        evaluation.py               Functions for model evaluation
        nii_dataset.py              The dataset class for .nii reading
```

# Setup

Run the following in command line to install the required packages:
```bash
pip install -r requirements.txt
```

# Usage

The evaluation script has specific requirements on the submission format. First, .nii volumes for each prediction and a .csv file containing the classification prediction are needed. .nii and .csv should be placed under the same directory as follows:
```
prediction_directory/
    RibFrac501.nii
    RibFrac502.nii
    ...
    RibFrac660.nii
    RibFrac_test_classification.csv
```
The ground-truth directory should follow exactly the same protocol:
```
ground_truth_directory/
    RibFrac501.nii
    RibFrac502.nii
    ...
    RibFrac660.nii
    RibFrac_test_classification.csv
```

Each .nii file should contain a 3D volume with ```n``` fracture regions labelled in integer from ```1``` to ```n```. The order of axes should be ```(x, y, z)```.

The classification prediction .csv should have four columns: ```pid``` (patient ID), ```label_index``` (prediction index in .nii volume), ```probs``` (detection probability) and ```class``` (fracture class), e.g.:

|pid|label_index|probs|class|
|-|-|-|-|
|RibFrac501|1|0.5|Buckle|
|RibFrac501|2|0.5|Displaced|
|...||||
|RibFrac660|2|0.5|Buckle|

Each row in the classification prediction csv represents one predicted fracture area. The pid should be in the same format as in .nii files. Please follow the exact directory and naming settings, or your submission won't be graded.

You can evaluate your model locally through the following command line:
```bash
python ribfrac/evaluation.py --gt_dir <ground_truth_directory> --pred_dir <prediction_directory>
