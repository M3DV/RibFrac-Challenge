# RibFrac-Challenge

**Note**: due to the change of expected file stucture, we are factoring the code. Please check the code later.

Setup and evaluation scripts for Rib Fracture Detection and Classification Challenge (RibFrac).



# Content

```
RibFract-Challenge/
    setup.py                        Initialize the directory and install the package
    ribfrac/
        decompression.py            Data decompression
        environ.py                  Record environment variables
        evaluation.py               Functions for model evaluation
        nii_dataset.py              The dataset class for .nii reading
```

# Setup

## Decompress data
To decompress the competition data, run the following command line:
```bash
python -m ribfrac.decompression --data_dir <custom/data/directory>
```
You need to specify the target data directory of your choice in argument ```data_dir```. This data directory will be written in ribfrac/environ.py, and can be accessed later once the package is installed. The decompression process takes 1.5-2 hours , and the decompressed data (train & val) should take up 277 GB.

The content structure of the decompressed data is as follows:
```
images/
    train/
        train_0.nii
        train_1.nii
        ...
    val/
        val_0.nii
        val_1.nii
        ...
labels/
    train/
        label_train_0.nii
        label_train_1.nii
        ...
    val/
        label_val_0.nii
        label_val_1.nii
        ...
```

## Install the RibFrac package
To install the RibFrac package, run:
```bash
python setup.py install
```
Now you can use the evaluation function by importing ```ribfrac```.

# Usage

## Evaluate model
You can evaluate your model through command line or package function.
### Command line
```bash
python -m ribfrac.evaluation --pred_dir <prediction/directory> --subset <{train, val}>
```
### Function call
```python
import ribfrac


subset = "val"      # choose between train or val
pred_dir = "prediction/val"
results, fpr, recall, auc = ribfrac.evaluation.evaluate(pred_dir, subset)
```
