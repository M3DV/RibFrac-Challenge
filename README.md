# RibFrac-Challenge
Setup and evaluation scripts for Rib Fracture Detection and Classification Challenge (RibFrac).

# Content
```
RibFract-Challenge/
    setup.py                    Initialize the data directory and decompress data
    decompression.py            Functions for data decompression
    evaluation.py               Functions for model evaluation
    base_array_generator.py     The abstract class for volume reading
```

# Setup

## Install required packages
With pip:
```
pip3 install -r requirements.txt
```
With Anaconda:
```
conda install --file requirements.txt
```
## Decompress data
To decompress all competition data, you need to specify the destination directory through command-line argument ```--data_dir```:
```
python setup.py --data_dir <your customed data directory>
```

# Usage

## Evaluate model
To evaluate your model, you first need to write your own array generator inheriting ```BaseArrayGenerator``` in ```base_array_generator.py``` and implement the ```__getitem__``` method. The ```BaseArrayGenerator``` class reads all files under ```root_dir``` and stores these paths under ```self.file_list```:

```python
class BaseArrayGenerator:
    """
    Abstract class of array generator. To evaluate your model predictions,
    please inherit this class and implement the __getitem__ method. The
    __getitem__ should return the prediction mask array corresponding to the
    index.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([os.path.join(root_dir, x)
            for x in os.listdir(root_dir)])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        raise NotImplementedError("Method __getitem__ not implemented.")
```

Your ```__getitem__``` method should take ```idx``` as argument, read the file at ```idx``` position, and return a ```numpy.ndarray``` containing 3D masks of ground-truth or prediction. For example:

```python
import numpy as np

from base_array_generator import BaseArrayGenerator


class MyArrayGenerator(BaseArrayGenerator):

    def __getitem__(self, idx):
        return np.load(self.file_list[idx])
```

Once your array generator is prepared, you can evaluate your model output using ```evaluation.evaluate```. This function will return the evaluation results for each prediction, plot the FROC curve and calculate the AUFROC score:

```python
# pseudo code
from evaluation import evaluate


pred_iter = MyArrayGenerator(pred_dir)
gt_iter = MyArrayGenerator(gt_dir)

eval_results, fpr, recall, aufroc = evaluate(pred_iter, gt_iter)
```