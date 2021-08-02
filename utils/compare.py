import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def compare_one_video(old_path, new_path, out_path):
    old_video = cv2.VideoCapture(old_path)
    new_video = cv2.VideoCapture(new_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (1280*2, 720))


    lengthold = int(old_video.get(cv2.CAP_PROP_FRAME_COUNT))
    lengthnew = int(new_video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(lengthold, lengthnew)

    length = min(lengthold, lengthnew)
    for _ in tqdm(range(length)):
        ret_old, old_frame = old_video.read()
        ret_new, new_frame = new_video.read()
        if not ret_old or not ret_new:
            break
        out_frame = np.concatenate([old_frame, new_frame], axis=1)
        out.write(out_frame)

    old_video.release()
    new_video.release()
    out.release()

pool = ThreadPool(8)

def mkdir_if_not_exist(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def thread_function(paths):
    old_path, new_path = paths
    print('Comparing {} and {}'.format(old_path, new_path))
    out_path = old_path.replace(old_folder, compared_folder)
    mkdir_if_not_exist(out_path)
    compare_one_video(old_path, new_path, out_path)

compared_folder  = '/home/ubuntu/tienpv/datasets/DMS_PnL_compared_negative_retina_mmdetection_resnet18/'

if __name__ == '__main__':
    old_folder = '/home/ubuntu/tienpv/datasets/DMS_PnL_retina_mmdetection_resnet18/'
    new_folder = '/home/ubuntu/tienpv/datasets/DMS_PnL_negative_retina_mmdetection_resnet18/'
    old_paths = glob(old_folder + '**/*.mp4', recursive=True)
    new_paths = glob(new_folder + '**/*.mp4', recursive=True)
    pool.map(thread_function, zip(old_paths, new_paths))
    # compare_one_video('test_03072021_unnormalize_paint.mp4', 'test_03072021_unnormalize_enhanced_data_paint.mp4', 'test_03072021_compared_paint.mp4')

    # for old_path, new_path in zip(old_paths, new_paths):
    #     print('Comparing {} and {}'.format(old_path, new_path))
    #     out_path = old_path.replace(old_folder, compared_folder)
    #     # mkdir_if_not_exist(out_path)
    #     compare_one_video(old_path, new_path, out_path)