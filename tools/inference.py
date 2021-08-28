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
def ravel_path(path, root):
    private_path = path[len(root):]
    return root + private_path.replace('/', '_')

def extract_file_list_fron_ann_file(root_dir, ann_file_path):
    image_paths = []
    f = open(ann_file_path,'r')
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            path = line[2:]
            image_paths.append(os.path.join(root_dir, path))
    return image_paths


config_file = '/home/ubuntu/tienpv/mmdetection/configs/retinanet/smoking_r50_fpn_1x_coco.py'
checkpoint_file = '/vinai/khaidq3/logs/mmdetection/logs/smoking_eating_drinking_split_logs/epoch_200.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# video
# video_dir = '/home/ubuntu/tienpv/datasets/DMS_PnL/'
# out_dir = '/home/ubuntu/tienpv/datasets/DMS_PnL_retina_mmdetection_resnet18/'
# # video_dir = '/home/ubuntu/tienpv/datasets/new_video/'
# # out_dir = '/home/ubuntu/tienpv/datasets/new_video_retina_mmdetection/'
# video_paths = glob.glob(video_dir+"*.mp4")
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# for video_path in tqdm(video_paths):
#     out_video_path = video_path.replace(video_dir, out_dir)
#     out_video_path = out_video_path.replace('webm', 'mp4')
#     out_video = cv2.VideoWriter(out_video_path, fourcc, 30, (1280, 720))
#     makesure_dir_is_exist(out_video_path)
#     video = mmcv.VideoReader(video_path)
#     for frame in tqdm(video):
#         result = inference_detector(model, frame)
#         out_frame = model.show_result(frame, result, bbox_color=(0,0,255))
#         out_video.write(out_frame)
#     out_video.release()

# image
image_dir = '/home/ubuntu/tienpv/datasets/smoking_eating_drinking/'
out_dir = '/home/ubuntu/tienpv/datasets/smoking_eating_drinking_split_out/'

# image_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) for f in filenames if f.endswith('.png')]

val_file_path = '/home/ubuntu/tienpv/datasets/smoking_eating_drinking/val/labels.txt'
image_paths = extract_file_list_fron_ann_file(image_dir, val_file_path)

for image_path in tqdm(image_paths):
    out_image_path = image_path.replace(image_dir, out_dir)
    out_image_path = ravel_path(out_image_path, out_dir)
    makesure_dir_is_exist(out_image_path)
    frame = cv2.imread(image_path)
    result = inference_detector(model, frame)
    out_image = model.show_result(frame, result, bbox_color=(0,0,255))
    cv2.imwrite(out_image_path, out_image)


val_file = ''
