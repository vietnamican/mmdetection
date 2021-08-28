import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SmokingDataset(CustomDataset):
    CLASSES = ('smoking', )

    def load_annotations(self, ann_file):
        data_infos = []
        f = open(ann_file,'r')
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
                    previous_bboxes.clear()
                    previous_labels.clear()
            else:
                line = line.split(' ')
                bbox = [float(x) for x in line]
                bbox = bbox[:4]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                previous_bboxes.append(bbox)
                previous_labels.append(0)
        
        previous_path = path
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