import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PhoneDataset(CustomDataset):
    CLASSES = ('phone', )
    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

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

        # for line in lines:
        #     line = line.rstrip()
        #     if line.startswith('#'):
        #         path = self.root_dir + line[2:]
        #         bboxes.clear()
        #         labels.clear()
        #     else:
        #         line = line.split(' ')
        #         bbox = [float(x) for x in line]
        #         bbox = bbox[:4]
        #         bbox[2] += bbox[0]
        #         bbox[3] += bbox[1]
        #         bboxes.append(bbox)
        #         labels.append(0)
        #         data_infos.append(
        #             dict(
        #                 filename=path,
        #                 width=1280,
        #                 height=720,
        #                 ann=dict(
        #                     bboxes=np.array(bboxes).astype(np.float32),
        #                     labels=np.array(labels).astype(np.int64)
        #                 )
        #             )
        #         )
        return data_infos
        

    # def load_annotations(self, ann_file):
    #     ann_list = mmcv.list_from_file(ann_file)

    #     data_infos = []
    #     for i, ann_line in enumerate(ann_list):
    #         if ann_line != '#':
    #             continue

    #         img_shape = ann_list[i + 2].split(' ')
    #         width = int(img_shape[0])
    #         height = int(img_shape[1])
    #         bbox_number = int(ann_list[i + 3])

    #         anns = ann_line.split(' ')
    #         bboxes = []
    #         labels = []
    #         for anns in ann_list[i + 4:i + 4 + bbox_number]:
    #             bboxes.append([float(ann) for ann in anns[:4]])
    #             labels.append(int(anns[4]))

    #         data_infos.append(
    #             dict(
    #                 filename=ann_list[i + 1],
    #                 width=width,
    #                 height=height,
    #                 ann=dict(
    #                     bboxes=np.array(bboxes).astype(np.float32),
    #                     labels=np.array(labels).astype(np.int64))
    #             ))

    #     return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']