import numpy as np

def _load_one_file_annotations(img_dir, ann_file):
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
                        img_dir=img_dir,
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
            img_dir=img_dir,
            ann=dict(
                bboxes=np.array(previous_bboxes).astype(np.float32),
                labels=np.array(previous_labels).astype(np.int64)
            )
        )
    )
    
    return data_infos

def load_annotations(img_dirs, ann_files):
    data_infos = []
    if isinstance(ann_files, str):
        data_infos = _load_one_file_annotations(img_dirs, ann_files)
    else:
        for img_dir, ann_file in zip(img_dirs, ann_files):
            data_infos.extend(_load_one_file_annotations(img_dir, ann_file))
    return data_infos