import os

import nibabel as nib
import numpy as np


class NiiDataset:
    """
    A dataloder reading all .nii files under one directory.

    Parameters
    ----------
    root_dir : str
        The directory where all .nii files reside.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = sorted([os.path.join(root_dir, x)
            for x in os.listdir(root_dir)
            if x.endswith(".nii") or x.endswith(".nii.gz")])
        self.pid_list = [os.path.basename(x).replace(".nii.gz", "")\
            .replace("-label", "") for x in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        array = nib.load(self.file_list[idx]).get_fdata()

        return array, self.pid_list[idx]
