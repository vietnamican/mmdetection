import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PhoneDataset(CustomDataset):
    CLASSES = ('phone', )

    def load_annotations(self, ann_file):
        f = open(ann_file,'r')
        data_infos = []
        lines = f.readlines()
        isFirst = True
        previous_bboxes = []
        previous_labels = []
        path = ''
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                previous_path = path
                path = line[2:]
                if isFirst:
                    isFirst = False
                else:
                    if len(previous_bboxes) == 0:
                        previous_bboxes = np.zeros((0, 4))
                        previous_labels = np.zeros(0)
                    data_infos.append(
                        dict(
                            filename=previous_path,
                            width=1280,
                            height=720,
                            ann=dict(
                                bboxes=np.array(previous_bboxes).astype(np.float32),
                                labels=np.array(previous_labels).astype(np.int64)
                            )
                        )
                    )
                    previous_bboxes = []
                    previous_labels = []
            else:
                line = line.split(' ')
                bbox = [float(x) for x in line]
                bbox = bbox[:4]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                previous_bboxes.append(bbox)
                previous_labels.append(0)
        
        previous_path = path
        if len(previous_bboxes) == 0:
            previous_bboxes = np.zeros((0, 4))
            previous_labels = np.zeros(0)
        data_infos.append(
            dict(
                filename=previous_path,
                width=1280,
                height=720,
                ann=dict(
                    bboxes=np.array(previous_bboxes).astype(np.float32),
                    labels=np.array(previous_labels).astype(np.int64)
                )
            )
        )

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']