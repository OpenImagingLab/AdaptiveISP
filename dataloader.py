import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed

import sys
sys.path.append("yolov3")
from yolov3.utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels
from yolov3.utils.general import LOGGER
from yolov3.utils.torch_utils import torch_distributed_zero_first
from yolov3.utils.dataloaders import seed_worker

from dataset import LoadImagesAndLabelsRAW, LoadImagesAndLabelsNormalize, \
    LoadImagesAndLabelsNormalizeHR, LoadImagesAndLabelsRAWHR, LoadImagesAndLabelsROD

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
WORLD_SIZE = 1


# def set_seed(seed=666):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)


def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      quad=False,
                      prefix='',
                      shuffle=False,
                      seed=0,
                      limit=-1,
                      add_noise=False,
                      brightness_range=None,
                      noise_level=None,
                      use_linear=False):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsRAW(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit,
            add_noise=add_noise,
            brightness_range=brightness_range,
            noise_level=noise_level,
            use_linear=use_linear,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabelsRAW.collate_fn4 if quad else LoadImagesAndLabelsRAW.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def create_dataloader_real(path,
                           imgsz,
                           batch_size,
                           stride,
                           single_cls=False,
                           hyp=None,
                           augment=False,
                           cache=False,
                           pad=0.0,
                           rect=False,
                           rank=-1,
                           workers=8,
                           image_weights=False,
                           quad=False,
                           prefix='',
                           shuffle=False,
                           seed=0,
                           limit=-1,
                           **kwargs):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsNormalize(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabelsNormalize.collate_fn4 if quad else LoadImagesAndLabelsNormalize.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def create_dataloader_rod(path,
                           imgsz,
                           batch_size,
                           stride,
                           single_cls=False,
                           hyp=None,
                           augment=False,
                           cache=False,
                           pad=0.0,
                           rect=False,
                           rank=-1,
                           workers=8,
                           image_weights=False,
                           quad=False,
                           prefix='',
                           shuffle=False,
                           seed=0,
                           limit=-1,
                           **kwargs):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsROD(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabelsROD.collate_fn4 if quad else LoadImagesAndLabelsROD.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def create_train_val_dataloader_real(path,
                                     imgsz,
                                     batch_size,
                                     stride,
                                     single_cls=False,
                                     hyp=None,
                                     augment=False,
                                     cache=False,
                                     pad=0.0,
                                     rect=False,
                                     rank=-1,
                                     workers=8,
                                     image_weights=False,
                                     quad=False,
                                     prefix='',
                                     shuffle=False,
                                     seed=0,
                                     limit=-1,
                                     **kwargs):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsNormalize(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit)

    batch_size = min(batch_size, len(dataset) // 2)
    # dataset = dataset[:len(dataset)//2 * 2]
    # train_dataset = dataset[:len(dataset) // 2]
    # val_dataset = dataset[len(dataset) // 2:]
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    n_train = len(dataset)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    train_dataloader = loader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and train_sampler is None,
                              num_workers=nw,
                              sampler=train_sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=LoadImagesAndLabelsNormalize.collate_fn4 if quad else LoadImagesAndLabelsNormalize.collate_fn,
                              worker_init_fn=seed_worker,
                              generator=generator)
    val_dataloader = loader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle and train_val_sampler is None,
                            num_workers=nw,
                            sampler=train_val_sampler,
                            pin_memory=PIN_MEMORY,
                            collate_fn=LoadImagesAndLabelsNormalize.collate_fn4 if quad else LoadImagesAndLabelsNormalize.collate_fn,
                            worker_init_fn=seed_worker,
                            generator=generator)
    return train_dataloader, val_dataloader, dataset


def create_dataloader_hr(path,
                        imgsz,
                        batch_size,
                        stride,
                        single_cls=False,
                        hyp=None,
                        augment=False,
                        cache=False,
                        pad=0.0,
                        rect=False,
                        rank=-1,
                        workers=8,
                        image_weights=False,
                        quad=False,
                        prefix='',
                        shuffle=False,
                        seed=0,
                        limit=-1,
                        add_noise=False,
                        brightness_range=None,
                        noise_level=None,
                        use_linear=False,
                        ):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsRAWHR(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit,
            add_noise=add_noise,
            brightness_range=brightness_range,
            noise_level=noise_level,
            use_linear=use_linear,
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabelsRAWHR.collate_fn4 if quad else LoadImagesAndLabelsRAWHR.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def create_dataloader_real_hr(path,
                            imgsz,
                            batch_size,
                            stride,
                            single_cls=False,
                            hyp=None,
                            augment=False,
                            cache=False,
                            pad=0.0,
                            rect=False,
                            rank=-1,
                            workers=8,
                            image_weights=False,
                            quad=False,
                            prefix='',
                            shuffle=False,
                            seed=0,
                            limit=-1,
                            **kwargs):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabelsNormalizeHR(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            limit=limit)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=LoadImagesAndLabelsNormalizeHR.collate_fn4 if quad else LoadImagesAndLabelsNormalizeHR.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def get_noise(batch_size, z_type="uniform", z_dim=27):
    if z_type == 'normal':
        return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
    elif z_type == 'uniform':
        return np.random.uniform(0, 1, [batch_size, z_dim]).astype(np.float32)
    else:
        assert False, 'Unknown noise type: %s' % z_type


def get_initial_states(batch_size, num_state_dim, filters_number):
    states = np.zeros(shape=(batch_size, num_state_dim), dtype=np.float32)
    for k in range(batch_size):
        for i in range(filters_number):
            # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
            # Used or not?
            # Initially nothing has been used
            states[k, -(i + 1)] = 0
    return states

