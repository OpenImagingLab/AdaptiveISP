import pickle as pickle
import os
import shutil
import datetime
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("yolov3")
from yolov3.utils.dataloaders import InfiniteDataLoader, LoadImagesAndLabels
from yolov3.utils.loss import ComputeLoss, ComputeLossBatch
from yolov3.utils.general import LOGGER, colorstr, intersect_dicts, check_dataset, TQDM_BAR_FORMAT, check_img_size, \
    labels_to_class_weights
from yolov3.utils.torch_utils import torch_distributed_zero_first
from yolov3.utils.dataloaders import seed_worker
from yolov3.utils.downloads import attempt_download
from yolov3.models.yolo import Model
from yolov3.utils.callbacks import Callbacks
from yolov3.utils.autoanchor import check_anchors
from yolov3.utils.metrics import fitness
from yolov3.utils.torch_utils import EarlyStopping
from yolov3.models.experimental import attempt_load

from replay_memory import ReplayMemory, create_input_tensor
from util import make_image_grid, Tee, merge_dict, Dict, save_img
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM
from agent import Agent
from value import Value
# from config import cfg
from dataloader import get_noise, get_initial_states, create_dataloader_real_hr


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
WORLD_SIZE = 1


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


class DynamicISP:
    def __init__(self, args, task="train_val"):
        train = False
        val = False
        if task == "train":
            train = True
        elif task == "train_val":
            train = True
            val = True
        if train:
            self.base_dir = os.path.join('experiments', args.save_path)
            os.makedirs(self.base_dir, exist_ok=True)
            self.log_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.tee = Tee(os.path.join(self.log_dir, 'log.txt'))
            self.writer = SummaryWriter(self.log_dir)
            self.image_dir = os.path.join(self.base_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)

            shutil.copy(f"config.py", os.path.join(self.base_dir, f"config.py"))
            print("Training begin....")
            print("------- Baseline + Truncated ---------")
        
        try:
            cfg = importlib.import_module(f'{args.cfg}').cfg
        except Exception as e:
            print(e)
            print(f"don't support {args.cfg}!")

        self.device = torch.device('cuda')
        cfg.filter_runtime_penalty = args.runtime_penalty
        cfg.filter_runtime_penalty_lambda = args.runtime_penalty_lambda

        # Hyperparameters
        hyp = args.hyp
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        args.hyp = hyp.copy()  # for saving hyps to checkpoints
        data_dict = check_dataset(args.data_cfg)
        nc = int(data_dict['nc'])  # number of classes
        resume = False

        # Load Pretrained YOLO
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(args.weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        yolo_model = Model(args.yolo_cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(
            self.device)  # create
        exclude = ['anchor'] if (args.yolo_cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, yolo_model.state_dict(), exclude=exclude)  # intersect
        yolo_model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(yolo_model.state_dict())} items from {weights}')  # report

        # Data Loader  TODO train
        train_path, val_path = data_dict['train'], data_dict['val']
        if task == "test":
            val_path = data_dict['test']
        # Image size
        gs = max(int(yolo_model.stride.max()), 32)  # grid size (max stride)
        args.imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
        # if train:
        self.train_loader = ReplayMemory(cfg, train, train_path, args.imgsz, args.batch_size, gs,
                                            single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                            rect=False, image_weights=False, prefix=colorstr('train: '), limit=-1,
                                            add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range, 
                                            noise_level=args.noise_level, use_linear=args.use_linear)
        if val:
            self.val_loader = ReplayMemory(cfg, val, val_path, args.imgsz, args.batch_size, gs,
                                           single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                           rect=False, image_weights=False, prefix=colorstr('val: '), limit=-1,
                                           add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                           noise_level=args.noise_level, use_linear=args.use_linear)
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)
        # Model attributes
        # check_anchors(dataset, model=yolo_model, thr=hyp['anchor_t'], imgsz=args.imgsz)  # run AutoAnchor
        nl = yolo_model.model[-1].nl  # number of detection layers (to scale hyps)  3
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = 0.0
        yolo_model.nc = nc  # attach number of classes to model
        yolo_model.hyp = hyp  # attach hyperparameters to model
        yolo_model.class_weights = labels_to_class_weights(self.train_loader.dataset.labels, nc).to(self.device) * nc  # attach class weights
        yolo_model.names = data_dict['names']  # class names
        yolo_model = yolo_model.to(self.device)
        self.yolo_model = yolo_model
        self.data_dict = data_dict

        self.agent = Agent(cfg, shape=(6 + len(cfg.filters), 64, 64)).to(self.device)
        self.value = Value(cfg, shape=(9 + len(cfg.filters), 64, 64)).to(self.device)
        self.args = args
        cfg.max_iter_step = int(self.args.epochs * 1000 // args.batch_size)  # 1000 train images
        if cfg.show_img_num > args.batch_size:
            cfg.show_img_num = args.batch_size

        self.gs = gs
        self.hyp = hyp
        self.val_path = val_path
        self.filter_name = [x.get_short_name() for x in self.agent.filters]

        print("----------------- args ------------------")
        for k, v in vars(args).items():
            print(k, ":", v)
        print("---------------- config ------------------")
        for k, v in cfg.items():
            print(k, ":", v)
        self.cfg = cfg

        self.max_bri = 0.9 # 0.8

    @staticmethod
    def compute_loss_batch(func, preds, targets, device):
        # for x in preds:
        #     print(x.shape)
        # print(targets_tensor.shape)
        batch = preds[0].shape[0]
        lclss = torch.zeros((batch, 1), device=device)  # class loss
        lboxs = torch.zeros((batch, 1), device=device)  # box loss
        lobjs = torch.zeros((batch, 1), device=device)  # object loss
        for b in range(batch):
            pred_one = []
            for i in range(len(preds)):
                pred_one.append(preds[i][b].unsqueeze(0).to(device))
            target_one = targets[b]
            target_one[:, 0] = 0
            # for x in pred_one:
            #     print(x.shape)
            # print(target_one.shape)
            lbox, lobj, lcls = func(pred_one, target_one.to(device))
            lboxs[b] = lbox
            lobjs[b] = lobj
            lclss[b] = lcls
        return lboxs + lobjs + lclss, torch.cat((lboxs, lobjs, lclss)).detach()

    def train(self):
        if self.args.resume is not None:
            print(f"Resume from {self.args.resume}")
            ckpt = torch.load(self.args.resume)
            self.agent.load_state_dict(ckpt['agent_model'])
            self.value.load_state_dict(ckpt['value_model'])

        agent_optimizer = torch.optim.Adam(self.agent.parameters(),
                                           lr=self.args.lr)  # , betas=(0.5, 0.9)
        value_optimizer = torch.optim.Adam(self.value.parameters(),
                                           lr=self.args.lr * float(self.cfg.value_lr_mul))  # , betas=(0.5, 0.9)
        lr_decay = 0.1
        # base_lr = self.args.lr
        # agent_lr_mul = 0.3
        segments = 3
        max_iter_step = self.cfg.max_iter_step
        agent_lr = lambda iter: lr_decay ** (1.0 * iter * segments / max_iter_step)
        value_lr = lambda iter: lr_decay ** (1.0 * iter * segments / max_iter_step)
        agent_scheduler = torch.optim.lr_scheduler.LambdaLR(agent_optimizer, lr_lambda=agent_lr)
        value_scheduler = torch.optim.lr_scheduler.LambdaLR(value_optimizer, lr_lambda=value_lr)
        # agent_scheduler = torch.optim.lr_scheduler.ExponentialLR(agent_optimizer, gamma=lr_decay, last_epoch=-1)
        # value_scheduler = torch.optim.lr_scheduler.ExponentialLR(value_optimizer, gamma=lr_decay, last_epoch=-1)
        print("'init learning rate, agent:", agent_scheduler.get_lr()[0], " value:", value_scheduler.get_lr()[0])

        compute_loss = ComputeLoss(self.yolo_model)
        compute_loss_batch = ComputeLossBatch(self.yolo_model, reduction="mean")  # sum
        callbacks = Callbacks()
        callbacks.run('on_train_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.args.workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.args.save_path)}\n"
                    f'Starting training for {0} epochs...')
        mloss_agent, mloss_value, mloss_detect = 0.0, 0.0, 0.0

        for iter in range(self.cfg.max_iter_step+1):
            self.agent.train()
            self.value.train()
            self.yolo_model.train()
            # fixed all layers
            for k, v in self.yolo_model.named_parameters():
                v.requires_grad = False
            for m in self.yolo_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            progress = float(iter) / self.cfg.max_iter_step
            feed_dict = self.train_loader.get_feed_dict_and_states(self.args.batch_size)
            # "im", "label", "path", "shape", "state", "z": numpy
            imgs, targets, paths, shapes, states = create_input_tensor(
                (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
            z = torch.from_numpy(feed_dict['z']).to(self.device)

            agent_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # callbacks.run('on_train_batch_start')
            imgs = imgs.to(self.device, non_blocking=True).float()  # input 0.0-1.0

            # Forward
            agent_out, agent_debug_out, agent_debugger = self.agent((imgs, z, states.to(self.device)), progress)
            retouch, new_states, surrogate, penalty = agent_out
            stopped = new_states[:, STATE_STOPPED_DIM:STATE_STOPPED_DIM + 1]

            pred_input = self.yolo_model(imgs)
            # detect_input_loss, detect_input_loss_items = compute_loss(pred_input, targets.to(self.device))  # loss scaled by batch_size
            detect_input_loss, _ = self.compute_loss_batch(compute_loss_batch, pred_input, feed_dict['label'], self.device)
            detect_input_loss = torch.clip(detect_input_loss * self.cfg.detect_loss_weight, 0, 1.0)

            pred_retouch = self.yolo_model(retouch)
            # detect_retouch_loss, detect_retouch_loss_items = compute_loss(pred_retouch, targets.to(self.device))  # loss scaled by batch_size
            _, detect_retouch_loss_items = compute_loss(pred_retouch, targets.to(self.device))  # loss scaled by batch_size
            detect_retouch_loss, _ = self.compute_loss_batch(compute_loss_batch, pred_retouch, feed_dict['label'], self.device)  # loss scaled by batch_size
            detect_retouch_loss = torch.clip(detect_retouch_loss * self.cfg.detect_loss_weight, 0, 1.0)

            reward = (self.cfg.all_reward + (1 - self.cfg.all_reward) * stopped) * \
                     (detect_input_loss.detach() - detect_retouch_loss) * self.cfg.critic_logit_multiplier
            # print("reward.shape", reward.shape, detect_input_loss.shape, detect_retouch_loss.shape)
            if self.cfg.use_penalty:
                reward -= penalty
            # print('new_states_slice', new_states)
            # print('new_states_slice', new_states[:, STATE_REWARD_DIM:STATE_REWARD_DIM + 1])
            # print('detect_retouch_loss shape', detect_retouch_loss.shape) [N, 1]

            old_value = self.value(imgs, states.to(self.device))
            new_value = self.value(retouch, new_states)

            clear_final = torch.gt(new_states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1], self.cfg.maximum_trajectory_length).float()
            new_value = new_value * (1.0 - clear_final)
            if self.args.use_truncated:
                retouch_mean = torch.mean(retouch, dim=(1, 2, 3)).unsqueeze(-1)
                truncated = torch.where(0.01 < retouch_mean, 1.0, 0.0)
                truncated = torch.where(retouch_mean < self.max_bri, truncated, torch.zeros_like(truncated))
                q_value = reward + (1.0 - stopped) * self.cfg.discount_factor * new_value * (1.0 - truncated)
            else:
                q_value = reward + (1.0 - stopped) * self.cfg.discount_factor * new_value
            advantage = q_value.detach() - old_value
            value_loss = torch.mean(advantage ** 2)  # , dim=(0, 1)

            # TD learning
            if self.cfg.use_TD:
                routine_loss = -q_value * self.cfg.parameter_lr_mul
                advantage = -advantage
            else:
                routine_loss = -reward
                advantage = -reward
            assert len(routine_loss.shape) == len(surrogate.shape)
            agent_loss = torch.mean(routine_loss + surrogate * advantage.detach())

            if iter % self.cfg.summary_freq == 0:
                try:
                    self.writer.add_scalar('agent_loss', agent_loss, global_step=iter)
                    self.writer.add_scalar('value_loss', value_loss, global_step=iter)
                    self.writer.add_scalar('detect_loss', detect_retouch_loss.mean(), global_step=iter)
                    self.writer.add_images('input', torch.clip(imgs[:self.cfg.show_img_num, ...], 0.0, 1.0), global_step=iter, dataformats="NCHW")
                    # self.writer.add_images('retouch', torch.clip(retouch[:self.cfg.show_img_num, ...], 0.0, 1.0), global_step=iter, dataformats="NCHW")
                except Exception as e:
                    print("write log error!")
                # get the selected filter name
                select_filter_name = []
                filter_id = list(agent_debug_out['selected_filter'].detach().cpu().numpy())
                for id_ in filter_id:
                    select_filter_name.append(self.filter_name[id_])
                out_image = torch.clip(retouch, 0.0, 1.0).detach().cpu().numpy()
                n, c, h, w = out_image.shape
                # out_image_res = np.zeros((n, h, w, c), dtype=np.float32)
                out_image_res = []
                for b in range(n):
                    tmp = np.transpose(out_image[b, ...], (1, 2, 0)).astype(np.float32).copy()
                    tmp = cv2.putText(tmp, select_filter_name[b], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
                    # out_image_res[b] = tmp
                    out_image_res.append(np.array(tmp))
                    # cv2.imshow('rgb', tmp)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                try:
                    # self.writer.add_images('retouch', out_image_res[:cfg.show_img_num, ...], global_step=iter, dataformats="NHWC")
                    self.writer.add_images('retouch', np.array(out_image_res[:self.cfg.show_img_num]), global_step=iter, dataformats="NHWC")
                except Exception as e:
                    print("write log error!")
                # print(old_value, new_value)

            # Backward
            value_loss.backward(retain_graph=False)
            agent_loss.backward(retain_graph=False)

            # clip gradient
            torch.nn.utils.clip_grad_norm(self.agent.parameters(), 1e-5)
            torch.nn.utils.clip_grad_norm(self.value.parameters(), 1e-5)

            agent_optimizer.step()
            value_optimizer.step()
            agent_scheduler.step()
            value_scheduler.step()

            mloss_agent = (mloss_agent * iter + agent_loss.item()) / (iter + 1)  # update mean losses
            mloss_value = (mloss_value * iter + value_loss.item()) / (iter + 1)  # update mean losses
            mloss_detect = (mloss_detect * iter + detect_retouch_loss_items.cpu().numpy()) / (iter + 1)  # update mean losses
            if iter % self.cfg.print_freq == 0:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      ('%11s' + '%8s,') % (f'{iter}/{self.cfg.max_iter_step - 1}', mem),
                      f"agent loss: {mloss_agent:.4f},",
                      f"value loss: {mloss_value:.4f},",
                      f"box_loss: {mloss_detect[0]:.4f}, obj_loss: {mloss_detect[1]:.4f}, cls_loss: {mloss_detect[2]:.4f},",
                      f"detect_retouch_loss: {detect_retouch_loss.mean().item():.2f},",
                      f"instances: {targets.shape[0]:2d},",
                      f"agent lr: {agent_scheduler.get_lr()[0]:.4e},",
                      f"value lr: {value_scheduler.get_lr()[0]:.4e},",
                      f"penalty: {penalty.mean().item():.4e}",
                      f"reward: {reward.mean().item():.4e}",
                )
                self.train_loader.debug()
            # callbacks.run('on_train_batch_end', self.a, ni, imgs, targets, paths, list(mloss))

            # update data pool
            if torch.isnan(retouch).any() or torch.isinf(retouch).any() or torch.mean(retouch) < 0.01 or torch.mean(retouch) > self.max_bri:
                print("retouch is nan or inf", torch.mean(retouch).detach().cpu().numpy())
                self.train_loader.fill_pool()
            else:
                self.train_loader.replace_memory(
                    self.train_loader.images_and_states_to_records(
                        retouch.detach().cpu().numpy(), feed_dict['label'], feed_dict['path'], feed_dict['shape'],
                        new_states.detach().cpu().numpy()))
            # validate
            if iter % self.cfg.val_freq == 0:
                self.agent.eval()
                self.yolo_model.eval()
                feed_dict = self.val_loader
                # "im", "label", "path", "shape", "state", "z": numpy
                imgs, targets, paths, shapes, states = create_input_tensor(
                    (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
                for b in range(imgs.shape[0]):
                    masks = []
                    decisions = []
                    operations = []
                    debug_info_list = []
                    retouch_img_trajs = []
                    retouch = imgs[b].unsqueeze(0).to(self.device)
                    retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                    noises = torch.from_numpy(np.array([self.train_loader.get_noise(1) for _ in range(self.cfg.test_steps)])).to(self.device)
                    states = torch.from_numpy(self.train_loader.get_initial_states(1)).to(self.device)
                    for i in range(self.cfg.test_steps):
                        (retouch, new_states, _, _), debug_info, generator_debugger = self.agent((retouch.float(), noises[i], states), 1.0)
                        retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                        states = new_states

                        debug_info_list.append(debug_info)
                        debug_plots = generator_debugger(debug_info, combined=False)
                        decisions.append(debug_plots[0])
                        operations.append(debug_plots[1])
                        masks.append(debug_plots[2])

                        save_img(retouch, paths[b], self.image_dir, f"{iter}_{i}")
                        if states[0][STATE_STOPPED_DIM] > 0:
                            break
                    padding = 4
                    patch = 64
                    grid = patch + padding
                    steps = len(retouch_img_trajs)

                    fused = np.ones(shape=(grid * 4, grid * steps, 3), dtype=np.float32)

                    for i in range(len(retouch_img_trajs)):
                        sx = grid * i
                        sy = 0
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            retouch_img_trajs[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)

                    for i in range(len(retouch_img_trajs) - 1):
                        sx = grid * i + grid // 2
                        sy = grid
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            decisions[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)
                        sy = grid * 2 - padding // 2
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            operations[i],
                            dsize=(patch, patch),
                            interpolation=cv2.INTER_NEAREST)
                        sy = grid * 3 - padding
                        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                            masks[i], dsize=(patch, patch), interpolation=cv2.INTER_NEAREST)

                    self.writer.add_image(f'val_{b}', fused, global_step=iter, dataformats="HWC")
                    # Save steps
                    save_img(fused, paths[b], self.image_dir, f"{iter}_steps", format="HWC")

                    # preds = self.yolo_model(retouch)
                    # # NMS
                    # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
                    # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                    # preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False,
                    #                             max_det=max_det)
                    # # Metrics
                    # for si, pred in enumerate(preds):
                    #     labels = targets[targets[:, 0] == si, 1:]
                    #     nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    #     path, shape = Path(paths[si]), shapes[si][0]
                    #     correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
                    #     seen += 1
                    #
                    #     predn = pred.clone()
                    #     scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
                    # # Plot images
                    # if plots and batch_i < 3:
                    #     plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                    #     plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
                    #                 names)  # pred

            if iter % self.cfg.save_model_freq == 0:
                self.agent.eval()
                self.value.eval()
                # Save model
                ckpt = {
                    'iter': iter,
                    'agent_model': self.agent.state_dict(),
                    'value_model': self.value.state_dict(),
                    # 'agent_scheduler': agent_scheduler.state_dict(),
                    # 'value_scheduler': value_scheduler.state_dict(),
                    'agent_optimizer': agent_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict(),
                }
                # Save last, best and delete
                torch.save(ckpt, os.path.join(self.ckpt_dir, f'DynamicISP_iter_{iter}.pth'))
                del ckpt
        torch.cuda.empty_cache()

    def val(self, batch_size=1, model_weights=None, steps=5):
        z_type = "uniform"
        z_dim = 16 + 3 + len(self.cfg.filters)
        filters_number = len(self.cfg.filters)
        num_state_dim = 3 + len(self.cfg.filters)

        base_dir = self.args.val_save_path
        os.makedirs(base_dir, exist_ok=True)
        image_dir = os.path.join(base_dir, "val-images")
        os.makedirs(image_dir, exist_ok=True)
        for i in range(steps):
            os.makedirs(os.path.join(image_dir, "step-"+str(i)), exist_ok=True)
        os.makedirs(os.path.join(image_dir, "all-step"), exist_ok=True)
        # callbacks = Callbacks()
        # callbacks.run('on_val_start')

        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.args.workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.args.val_save_path)}\n"
                    f'Starting eval ...')

        self.agent.load_state_dict(torch.load(model_weights)['agent_model'])
        self.agent.eval()
        self.yolo_model.eval()

        if self.args.data_name in ("lod", ):
            self.val_loader, _ = create_dataloader_real_hr(self.val_path, self.args.imgsz, batch_size, self.gs, False,
                                                           hyp=self.hyp, cache=False, rect=False, workers=1, pad=0.0,
                                                           prefix=colorstr('val: '), add_noise=self.args.add_noise)
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        pbar = tqdm(self.val_loader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for batch_i, (imgs, targets, paths, shapes, imgs_hr) in enumerate(pbar):
        # self.train_loader.load()
        # feed_dict = self.train_loader.get_feed_dict_and_states(8)
        # imgs, targets, paths, shapes, states = create_input_tensor(
        #     (feed_dict['im'], feed_dict['label'], feed_dict['path'], feed_dict['shape'], feed_dict['state']))
        # for b in range(imgs.shape[0]):
            # callbacks.run('on_val_batch_start')
            masks = []
            decisions = []
            operations = []
            debug_info_list = []
            retouch_img_trajs = []
            retouch = imgs.to(self.device)
            retouch_hr = imgs_hr.to(self.device)
            # retouch = imgs[b].unsqueeze(0).to(self.device)
            retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))

            nb = imgs.shape[0]
            noises = torch.from_numpy(np.array([get_noise(nb, z_type, z_dim) for _ in range(steps)])).to(self.device)
            states = torch.from_numpy(get_initial_states(nb, num_state_dim, filters_number)).to(self.device)
            for i in range(steps):
                (retouch, new_states, retouch_hr), debug_info, generator_debugger = self.agent((retouch.float(), noises[i], states), 1.0, retouch_hr.float())
                retouch_img_trajs.append(np.transpose(retouch[0].detach().cpu().numpy(), (1, 2, 0)))
                states = new_states

                debug_info_list.append(debug_info)
                debug_plots = generator_debugger(debug_info, combined=False)
                decisions.append(debug_plots[0])
                operations.append(debug_plots[1])
                masks.append(debug_plots[2])

                # save_img(retouch_hr, paths[0], image_dir, f"{i}")
                save_img(retouch_hr, paths[0], os.path.join(image_dir, "step-" +str(i)), None, "CHW", False)
                if states[0][STATE_STOPPED_DIM] > 0:
                    break
            padding = 4
            patch = 64
            grid = patch + padding
            steps = len(retouch_img_trajs)

            fused = np.ones(shape=(grid * 4, grid * steps, 3), dtype=np.float32)

            for i in range(len(retouch_img_trajs)):
                sx = grid * i
                sy = 0
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    retouch_img_trajs[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)

            for i in range(len(retouch_img_trajs) - 1):
                sx = grid * i + grid // 2
                sy = grid
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    decisions[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)
                sy = grid * 2 - padding // 2
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    operations[i],
                    dsize=(patch, patch),
                    interpolation=cv2.INTER_NEAREST)
                sy = grid * 3 - padding
                fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
                    masks[i], dsize=(patch, patch), interpolation=cv2.INTER_NEAREST)

            # Save steps
            # save_img(fused, paths[0], image_dir, f"steps", format="HWC")
            save_img(fused, paths[0], os.path.join(image_dir, "all-step"), None, "HWC", False)

            # preds = self.yolo_model(retouch)
            # # NMS
            # targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
            # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            # preds = non_max_suppression(preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=False,
            #                             max_det=max_det)
            # # Metrics
            # for si, pred in enumerate(preds):
            #     labels = targets[targets[:, 0] == si, 1:]
            #     nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            #     path, shape = Path(paths[si]), shapes[si][0]
            #     correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            #     seen += 1
            #
            #     predn = pred.clone()
            #     scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            # # Plot images
            # if plots and batch_i < 3:
            #     plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            #     plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg',
            #                 names)  # pred
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='train_val', help="train, train and val, val")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=800, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=20, help="scheduler_step_size")
    parser.add_argument("--scheduler_lr_gamma", type=float, default=0.5, help="scheduler_lr_gamma")
    parser.add_argument("--imgsz", type=int, default=512, help="image size")
    parser.add_argument("--workers", type=int, default=4, help="workers")
    
    parser.add_argument('--weights', type=str, default='../../pretrained/yolov3.pt', help='yolov3 pretrained path')
    parser.add_argument('--yolo_cfg', type=str, default='yolov3/models/yolov3.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='yolov3/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument("--save_path", type=str, default='adaptiveisp', help="save path at experiments/save_path/")
    parser.add_argument("--data_name", type=str, default='lod', choices=['lod', ], help="train data: lod")
    parser.add_argument('--data_cfg', type=str, default='yolov3/data/lod.yaml', help='dataset.yaml path')
    parser.add_argument("--add_noise", type=bool, default=False, help="add_noise")
    parser.add_argument("--use_linear", action='store_true', default=False, help="use linear noise distribution")
    parser.add_argument("--bri_range", type=float, default=None, nargs='*', help="brightness range, (low, high), 0.0~1.0")
    parser.add_argument("--noise_level", type=float, default=None, help="noise_level, 0.001~0.012")

    parser.add_argument('--use_truncated', type=bool, default=True, help='use_truncated')
    parser.add_argument("--runtime_penalty", action='store_true', default=False, help="use runtime penalty")
    parser.add_argument("--runtime_penalty_lambda", type=float, default=0.01, help="use runtime penalty lambda")
    parser.add_argument('--resume', type=str, default=None, help='resume model weights')

    parser.add_argument('--model_weights', type=str, default='experiments/', help='isp model weight')
    parser.add_argument("--val_save_path", type=str, default='experiments/adaptiveisp')
    parser.add_argument("--steps", type=int, default=5, help="steps")
    parser.add_argument("--cfg", type=str, default="config", help="config py file")

    args = parser.parse_args()
    args.save_path = args.data_name + '-' + args.save_path
    if args.data_name in ("lod", ):
        args.add_noise = False
        args.bri_range = None
        args.use_linear = False

    Task = DynamicISP(args, args.task)
    if args.task == "train" or args.task == "train_val":
        Task.train()
    elif args.task == "val":
        Task.val(model_weights=args.model_weights, steps=args.steps)