import os


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
