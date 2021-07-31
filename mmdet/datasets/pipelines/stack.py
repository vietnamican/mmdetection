import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES


@PIPELINES.register_module()
class Stack:

    def __init__(self):
        pass

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img = results['img'][..., np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
        results['img'] = img
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'stack grayscale image into three channels')
        return repr_str