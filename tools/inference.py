import os
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import glob
from tqdm import tqdm 

def makesure_dir_is_exist(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

config_file = '/home/ubuntu/tienpv/mmdetection/configs/retinanet/phone_r18_fpn_1x_coco.py'
checkpoint_file = '/vinai/khaidq3/logs/mmdetection/resnet18_copy_and_paste_logs/epoch_200.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

video_dir = '/home/ubuntu/tienpv/datasets/DMS_PnL/'
out_dir = '/home/ubuntu/tienpv/datasets/DMS_PnL_copy_and_paste_retina_mmdetection_resnet18/'
# video_dir = '/home/ubuntu/tienpv/datasets/new_video/'
# out_dir = '/home/ubuntu/tienpv/datasets/new_video_retina_mmdetection/'
video_paths = glob.glob(video_dir+"*.mp4")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
for video_path in tqdm(video_paths):
    out_video_path = video_path.replace(video_dir, out_dir)
    out_video_path = out_video_path.replace('webm', 'mp4')
    out_video = cv2.VideoWriter(out_video_path, fourcc, 30, (1280, 720))
    makesure_dir_is_exist(out_video_path)
    video = mmcv.VideoReader(video_path)
    for frame in tqdm(video):
        result = inference_detector(model, frame)
        out_frame = model.show_result(frame, result, bbox_color=(0,0,255))
        out_video.write(out_frame)
    out_video.release()
