import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
from .phone_dataset import PhoneDataset

def is_apply_copy_and_paste():
    return np.random.rand() < 0.5

def choose_randomly_index(upper_bound):
    return np.random.randint(upper_bound)

def is_empty_gt(sample):
    return sample['ann']['bboxes'].shape[0] == 0

@DATASETS.register_module()
class CopyAndPasteDataset(PhoneDataset):
    def __init__(self, ann_file):
        super().__init__(ann_file)
    
    def load_annotations(self, ann_file):
        data_infos = []
        for data_info in self.data_infos:
            if is_empty_gt(data_info):
                if is_apply_copy_and_paste():
                    index = -1
                    while True:
                        index = choose_randomly_index(self.__length__)
                        if not is_empty_gt(self.data_infos[index]):
                            break
                    