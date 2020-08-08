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

You can evaluate your model through the following command line:
```bash
python ribfrac/evaluation.py --gt_dir <ground_truth/directory> --pred_dir <prediction/directory>
