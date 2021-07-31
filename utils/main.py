from tqdm import tqdm
import glob

from mmdet.apis import init_detector, inference_detector

# model.show_result(img, result, out_file='demo/result.jpg')

config_path = '/home/ubuntu/tienpv/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
checkpoint_path = '/home/ubuntu/tienpv/mmdetection/checkpoints/retinanet_r50_fpn_1x_coco.pth'

model = init_detector(config_path, checkpoint_path, device='cuda:0')
img_dir = '/home/ubuntu/tienpv/datasets/ourDB/images'
imgs = glob.glob(img_dir + '/*.png', recursive=True)
# results = inference_detector(model, imgs)
for img in tqdm(imgs):
    result = inference_detector(model, img)
    out = img.replace('images', 'results')
    model.show_result(img, result, out_file=out)