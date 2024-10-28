import cv2
import numpy as np
import torch
import random
import os
import hashlib
import math
import gzip

import sys
sys.path.append("yolov3")

from yolov3.utils.dataloaders import LoadImagesAndLabels
from yolov3.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                        letterbox, mixup, random_perspective)
from yolov3.utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                                  check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                                  xywh2xyxy, xywhn2xyxy, xyxy2xywhn)

from isp.unprocess_np import unprocess_wo_mosaic
from util import AsyncTaskManager

from multiprocessing.pool import Pool, ThreadPool
from tqdm import tqdm
from pathlib import Path
from yolov3.utils.dataloaders import img2label_paths, get_hash
import glob
from itertools import repeat
HELP_URL = 'See https://docs.ultralytics.com/yolov5/tutorials/train_custom_data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', "npy"  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders


# COCO unprocess
class LoadImagesAndLabelsRAW(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAW, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        if not self.train:
            seed = int(os.path.splitext(os.path.split(self.im_files[index])[1])[0])
            np.random.seed(seed)
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class LoadImagesAndLabelsRAWV2(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWV2, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        if not self.train:
            seed = int(os.path.splitext(os.path.split(self.im_files[index])[1])[0])
            np.random.seed(seed)
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB
        img = (img * 65535).astype(np.uint16)

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        img = img.astype(np.float32) / 65535.
        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class LoadImagesAndLabelsRAWHR(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                     cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.use_linear = use_linear
        self.train = True if 'train' in prefix else False

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        if not self.train:
            seed = int(os.path.splitext(os.path.split(self.im_files[index])[1])[0])
            np.random.seed(seed)
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB
        img_hr = img.copy()

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        img_hr = img_hr.transpose((2, 0, 1))  # HWC to CHW
        img_hr = np.ascontiguousarray(img_hr)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes, img_hr = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)
    
    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

    # use original image, but OOM
    # def load_image(self, i):
    #     # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #             assert im is not None, f'Image Not Found {f}'
    #         h0, w0 = im.shape[:2]  # orig hw
    #         # r = self.img_size / max(h0, w0)  # ratio
    #         # if r != 1:  # if sizes are not equal
    #         #     interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    #         #     im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    #         return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


class LoadImagesAndLabelsRAWReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1,
                 add_noise=False, 
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False):
        super(LoadImagesAndLabelsRAWReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.use_linear = use_linear

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        img = img[..., ::-1]  # BGR to RGB uint8
        img = img / 255.0  # uint8 to float

        # rgb_img = img
        # rgb_img = rgb_img.transpose((2, 0, 1))  # HWC to CHW
        # TODO add unprocess
        img, _ = unprocess_wo_mosaic(img, self.add_noise, self.brightness_range, self.noise_level, self.use_linear)  # RGB to linear RGB

        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, color=(0, 0, 0), auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            img, labels = random_perspective(img,
                                             labels,
                                             degrees=hyp['degrees'],
                                             translate=hyp['translate'],
                                             scale=hyp['scale'],
                                             shear=hyp['shear'],
                                             perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        # raw img: 0-1
        # labels_out = [n, 6]
        # file path
        # shapes: (h0, w0), ((h / h0, w / w0), pad)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)
        # return self.collate_fn_raw([im_list, label_list, path_list, shapes_list])
        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret


# LOD OPRD
class LoadImagesAndLabelsNormalize(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalize, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes


class LoadImagesAndLabelsNormalizeHR(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeHR, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                           cache_images, single_cls, stride, pad, min_items, prefix, limit)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            img_hr = img.copy()

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        img_hr = img_hr.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_hr = np.ascontiguousarray(img_hr) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes, torch.from_numpy(img_hr)
    
    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes, img_hr = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes, torch.stack(img_hr, 0)

    # use original image, but OOM
    # def load_image(self, i):
    #     # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
    #     im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
    #     if im is None:  # not cached in RAM
    #         if fn.exists():  # load npy
    #             im = np.load(fn)
    #         else:  # read image
    #             im = cv2.imread(f)  # BGR
    #             assert im is not None, f'Image Not Found {f}'
    #         h0, w0 = im.shape[:2]  # orig hw
    #         # r = self.img_size / max(h0, w0)  # ratio
    #         # if r != 1:  # if sizes are not equal
    #         #     interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
    #         #     im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    #         return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    #     return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized


class LoadImagesAndLabelsNormalizeReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        super(LoadImagesAndLabelsNormalizeReplay, self).__init__(path, img_size, batch_size, augment, hyp, rect, image_weights,
                                                                 cache_images, single_cls, stride, pad, min_items, prefix, limit)
        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) / 255.

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)
        # return self.collate_fn_raw([im_list, label_list, path_list, shapes_list])
        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret


def img2label_paths_rod(img_paths, img_dir_name="images"):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}{img_dir_name}{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabelsRODReplay(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        img_dir_name = "npy"
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        img_dir_name = t[0].split("/")[1] # ./images/*.png, get the images name
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        if 0 < limit < len(self.im_files):
            self.im_files = self.im_files[:limit]
            LOGGER.warning(f"Select {limit} images as training data!")

        # Check cache
        self.label_files = img2label_paths_rod(self.im_files, img_dir_name)  # labels

        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training.'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training.'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    @staticmethod
    def verify_image_label(args):
        # Verify one image-label pair
        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        try:
            # verify images
            im = np.load(im_file)
            shape = im.shape
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            # assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            # if im.format.lower() in ('jpg', 'jpeg'):
            #     with open(im_file, 'rb') as f:
            #         f.seek(-2, 2)
            #         if f.read() != b'\xff\xd9':  # corrupt JPEG
            #             ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
            #             msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)
            # print(im_file, lb, shape, segments, nm, nf, ne, nc, msg)
            # exit()
            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {path.parent / path.stem}...'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(self.verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            # print("fn.exists()", fn.exists())
            if fn.exists():  # load npy  True
                im = np.load(fn)
                im = im / np.percentile(im, 99) # TODO
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1] # # bgr
            else:  # read image
                # im = cv2.imread(f)  # BGR
                im = np.load(f)
                im = im / np.percentile(im, 99) # TODO
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1] # bgr
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # print(img.min(), img.max()) # HDR data, have *255 as input

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn_raw(batch):
        im, label, path, shapes = batch  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        im_list = []
        label_list = []
        path_list = []
        shapes_list = []
        for i in range(len(batch)):
            im, label, path, shapes = self.__getitem__(batch[i])
            im_list.append(im)
            label_list.append(label)
            path_list.append(path)
            shapes_list.append(shapes)
        # return self.collate_fn_raw([im_list, label_list, path_list, shapes_list])
        return im_list, label_list, path_list, shapes_list

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(target=self.get_next_batch_, args=(self.default_batch_size,))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret


class LoadImagesAndLabelsROD(LoadImagesAndLabels):
    def __init__(self,
                 path,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 rect=False,
                 image_weights=False,
                 cache_images=False,
                 single_cls=False,
                 stride=32,
                 pad=0.0,
                 min_items=0,
                 prefix='',
                 limit=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations(size=img_size) if augment else None

        img_dir_name = "npy"
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        img_dir_name = t[0].split("/")[1] # ./images/*.png, get the images name
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent, 1) if x.startswith('./') else x for x in t]  # to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n') from e

        if 0 < limit < len(self.im_files):
            self.im_files = self.im_files[:limit]
            LOGGER.warning(f"Select {limit} images as training data!")

        # Check cache
        self.label_files = img2label_paths_rod(self.im_files, img_dir_name)  # labels

        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training.'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training.'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update

        # Filter images
        if min_items:
            include = np.array([len(x) >= min_items for x in self.labels]).nonzero()[0].astype(int)
            LOGGER.info(f'{prefix}{n - len(include)}/{n} images filtered from dataset')
            self.im_files = [self.im_files[i] for i in include]
            self.label_files = [self.label_files[i] for i in include]
            self.labels = [self.labels[i] for i in include]
            self.segments = [self.segments[i] for i in include]
            self.shapes = self.shapes[include]  # wh

        # Create indices
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        self.segments = list(self.segments)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = [segment[idx] for idx, elem in enumerate(j) if elem]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into RAM/disk for faster training
        if cache_images == 'ram' and not self.check_cache_ram(prefix=prefix):
            cache_images = False
        self.ims = [None] * n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
            self.im_hw0, self.im_hw = [None] * n, [None] * n
            fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image
            results = ThreadPool(NUM_THREADS).imap(fcn, range(n))
            pbar = tqdm(enumerate(results), total=n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache_images == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({b / gb:.1f}GB {cache_images})'
            pbar.close()

        self.synchronous = False
        self.default_batch_size = 64
        self.async_task = None
        self.num_images = len(self.shapes)

    @staticmethod
    def verify_image_label(args):
        # Verify one image-label pair
        im_file, lb_file, prefix = args
        nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
        try:
            # verify images
            im = np.load(im_file)
            shape = im.shape
            assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
            # assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
            # if im.format.lower() in ('jpg', 'jpeg'):
            #     with open(im_file, 'rb') as f:
            #         f.seek(-2, 2)
            #         if f.read() != b'\xff\xd9':  # corrupt JPEG
            #             ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
            #             msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

            # verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                    assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0, 5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0, 5), dtype=np.float32)
            # print(im_file, lb, shape, segments, nm, nf, ne, nc, msg)
            # exit()
            return im_file, lb, shape, segments, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
            return [None, None, None, None, nm, nf, ne, nc, msg]

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{prefix}Scanning {path.parent / path.stem}...'
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(self.verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc=desc,
                        total=len(self.im_files),
                        bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i],
        if im is None:  # not cached in RAM
            # print("fn.exists()", fn.exists())
            if fn.exists():  # load npy  True
                im = np.load(fn).astype(np.float32)
                im = im / np.percentile(im, 99) # TODO
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1] # # bgr
            else:  # read image
                # im = cv2.imread(f)  # BGR
                im = np.load(f).astype(np.float32)
                im = im / np.percentile(im, 99) # TODO
                im = np.clip(im, 0.0, 1.0)
                im = im[::-1] # bgr
                assert im is not None, f'Image Not Found {f}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def __getitem__(self, index):
        # TODO must comment this, just use input index
        # index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment, color=(0, 0, 0))
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # print(img.min(), img.max()) # HDR data, have *255 as input

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes


def restore_image(image, ori_image):
    ih, iw, _ = image.shape
    if isinstance(ori_image, (tuple, list)):
        h, w, _ = ori_image
    else:
        h, w, _ = ori_image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    dst_img = image[dh:dh + nh, dw:dw + nw, ::]
    # print(dst_img.shape)
    dst_img = cv2.resize(dst_img, (w, h))
    # print(dst_img.shape)
    # print(scale, dw, dh, nw, nh)
    return dst_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def show(x, title="a", format="HWC", is_last=True):
        if format == 'CHW':
            x = np.transpose(x, (1, 2, 0))
        plt.figure()
        plt.cla()
        plt.title(title)
        plt.imshow(x)
        if is_last:
            plt.show()
    data_dict = {'path': 'COCO/coco2017',
     'train': 'COCO/coco2017/train2017.txt',
     'val': 'COCO/coco2017/val2017.txt',
     'test': 'COCO/coco2017/test-dev2017.txt',
     'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
               8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
               14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
               22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
               29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
               35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
               40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
               48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
               55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
               62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
               69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
               76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'},
     'download': "from utils.general import download, Path\n\n\n# Download labels\nsegments = False  # segment or box labels\ndir = Path(yaml['path'])  # dataset root dir\nurl = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'\nurls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels\ndownload(urls, dir=dir.parent)\n\n# Download data\nurls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images\n        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images\n        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)\ndownload(urls, dir=dir / 'images', threads=3)\n",
     'nc': 80}

    batch_size = 4
    imgsz = 512
    val_path = 'COCO/coco2017/val2017.txt'
    dataset = LoadImagesAndLabelsRAWReplay(
        val_path,
        imgsz,
        batch_size,
        augment=False,  # augmentation
        limit=1000
    )
    print(len(dataset))
    print(dataset.get_next_batch_(batch_size))
    exit()



