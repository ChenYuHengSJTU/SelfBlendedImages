import cv2
import numpy as np
import torch
from PIL import Image
import albumentations as alb
import random
import logging

import sys
sys.path.append('/home/chenyuheng/SelfBlendedImages/src')


from utils import blend as B
from utils.funcs import IoUfrom2bboxes, crop_face, RandomDownScale


def get_source_transforms():
    """获取源图像的转换"""
    return alb.Compose([
        alb.Compose([
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
            alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
        ],p=1),
        alb.OneOf([
            RandomDownScale(p=1),
            alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        ],p=1),
    ], p=1.)

def get_transforms():
    """获取图像转换"""
    return alb.Compose([
        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
        alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
    ], 
    additional_targets={'image1': 'image'},
    p=1.)

def randaffine(img, mask):
    """随机仿射变换"""
    f = alb.Affine(
        translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
        scale=[0.95,1/0.95],
        fit_output=False,
        p=1)
    
    g = alb.ElasticTransform(
        alpha=50,
        sigma=7,
        alpha_affine=0,
        p=1,
    )

    transformed = f(image=img, mask=mask)
    img = transformed['image']
    mask = transformed['mask']
    transformed = g(image=img, mask=mask)
    mask = transformed['mask']
    return img, mask

def reorder_landmark(landmark):
    """重新排序关键点"""
    landmark_add = np.zeros((13,2))
    for idx, idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark

def hflip(img, mask=None, landmark=None, bbox=None):
    """水平翻转"""
    H,W = img.shape[:2]
    landmark = landmark.copy()
    bbox = bbox.copy()

    if landmark is not None:
        landmark_new = np.zeros_like(landmark)
        
        # 关键点重排序逻辑...
        landmark_pairs = [
            (slice(0,17), slice(0,17)),
            (slice(17,27), slice(17,27)),
            (slice(27,31), slice(27,31)),
            (slice(31,36), slice(31,36)),
            (slice(36,40), slice(42,46)),
            (slice(40,42), slice(46,48)),
            (slice(42,46), slice(36,40)),
            (slice(46,48), slice(40,42)),
            (slice(48,55), slice(48,55)),
            (slice(55,60), slice(55,60)),
            (slice(60,65), slice(60,65)),
            (slice(65,68), slice(65,68))
        ]
        
        for dst, src in landmark_pairs:
            landmark_new[dst] = landmark[src][::-1]
            
        if len(landmark) == 81:
            landmark_new[68:81] = landmark[68:81][::-1]
            
        landmark_new[:,0] = W - landmark_new[:,0]
    else:
        landmark_new = None

    if bbox is not None:
        bbox_new = np.zeros_like(bbox)
        bbox_new[0,0] = bbox[1,0]
        bbox_new[1,0] = bbox[0,0]
        bbox_new[:,0] = W - bbox_new[:,0]
        bbox_new[:,1] = bbox[:,1].copy()
        if len(bbox) > 2:
            bbox_new[2:7] = bbox[2:7].copy()
            bbox_new[2:7,0] = W - bbox_new[2:7,0]
    else:
        bbox_new = None

    if mask is not None:
        mask = mask[:,::-1]
    
    img = img[:,::-1].copy()
    return img, mask, landmark_new, bbox_new

def get_item(idx, image_list, image_size=224, phase='train', path_lm='/landmarks/'):
    """
    获取数据集中的一个样本
    
    参数:
        idx: 样本索引
        image_list: 图像文件路径列表
        image_size: 输出图像大小
        phase: 训练阶段 ('train', 'val', 'test')
        path_lm: 关键点文件路径
    """
    transforms = get_transforms()
    source_transforms = get_source_transforms()
    image_size = (image_size, image_size)
    
    flag = True
    while flag:
        try:
            filename = image_list[idx]
            img = np.array(Image.open(filename))
            landmark = np.load(filename.replace('.png','.npy').replace('/frames/',path_lm))[0]
            bbox_lm = np.array([landmark[:,0].min(), landmark[:,1].min(), 
                              landmark[:,0].max(), landmark[:,1].max()])
            
            bboxes = np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
            iou_max = -1
            for i in range(len(bboxes)):
                iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                if iou_max < iou:
                    bbox = bboxes[i]
                    iou_max = iou

            landmark = reorder_landmark(landmark)
            if phase == 'train':
                if np.random.rand() < 0.5:
                    img, _, landmark, bbox = hflip(img, None, landmark, bbox)
                    
            img, landmark, bbox, _ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False)

            # 自混合处理
            H,W = len(img), len(img[0])
            if np.random.rand() < 0.25:
                landmark = landmark[:68]
                
            mask = np.zeros_like(img[:,:,0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

            source = img.copy()
            if np.random.rand() < 0.5:
                source = source_transforms(image=source.astype(np.uint8))['image']
            else:
                img = source_transforms(image=img.astype(np.uint8))['image']

            source, mask = randaffine(source, mask)
            img_blended, mask = B.dynamic_blend(source, img, mask)
            img_blended = img_blended.astype(np.uint8)
            img = img.astype(np.uint8)

            if phase == 'train':
                transformed = transforms(image=img_blended, image1=img)
                img_f = transformed['image']
                img_r = transformed['image1']
            else:
                img_f = img_blended
                img_r = img

            img_f, _, _, _, y0_new, y1_new, x0_new, x1_new = crop_face(
                img_f, landmark, bbox, margin=False, crop_by_bbox=True, 
                abs_coord=True, phase=phase
            )
            
            img_r = img_r[y0_new:y1_new, x0_new:x1_new]
            
            img_f = cv2.resize(img_f, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
            img_r = cv2.resize(img_r, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
            
            img_f = img_f.transpose((2,0,1))
            img_r = img_r.transpose((2,0,1))
            flag = False
            
        except Exception as e:
            print(e)
            idx = random.randint(0, len(image_list)-1)
    
    return img_f, img_r
# 使用示例
# image_list = ['/path/to/image1.png', '/path/to/image2.png', ...]
image_list = ['/dataset/FaceForensics++/original_sequences/youtube/raw/frames/039/353.png', '/dataset/FaceForensics++/original_sequences/youtube/raw/frames/039/431.png', '/dataset/FaceForensics++/original_sequences/youtube/raw/frames/039/529.png', '/dataset/FaceForensics++/original_sequences/youtube/raw/frames/039/608.png']

for i in range(len(image_list)):
    img_f, img_r = get_item(
        idx=i, 
        image_list=image_list,
        image_size=224,
        phase='train',
        path_lm='/landmarks/'
    )