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

Each .nii file should contain a 3D volume with ```n``` fracture regions labelled in integer from ```1``` to ```n```. The order of axes should be ```(x, y, z)```.

The classification prediction .csv should have four columns: ```public_id``` (patient ID), ```label_id``` (prediction ID marking the specific connected-region in the .nii volume), ```confidence``` (detection confidence) and ```label_code``` (fracture class), e.g.:

|public_id|label_id|confidence|label_code|
|-|-|-|-|
|RibFrac501|0|0.5|0|
|RibFrac501|1|0.5|1|
|RibFrac501|2|0.5|2|
|...||||
|RibFrac660|0|0.5|0|
|RibFrac660|1|0.5|3|

For each public_id, there should at least be one row representing the background class. As in the ground-truth info .csv, the background record should have ```label_id=0``` and ```label_code=0```. Other than that, each row in the classification prediction .csv represents one predicted fracture area. The public_id should be in the same format as in .nii file names. Please follow the exact directory and naming settings, or your submission won't be graded.

You can evaluate your model locally through the following command line:
```bash
python ribfrac/evaluation.py --gt_dir <ground_truth_directory> --pred_dir <prediction_directory>
```
