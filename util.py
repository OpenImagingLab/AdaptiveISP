import math
import cv2
import torch
import os
import sys
import numpy as np
import threading
'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3


def save_img(img, img_path, save_path, prefix=None, format="CHW", is_train=True):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # print(img.shape, len(img.shape))
    if len(img.shape) > 3:
        img = img.squeeze(0)
    if format.upper() == "CHW":
        img = np.transpose(img, (1, 2, 0))
    img[np.isnan(img)] = 0.
    # print(img.shape, format)
    img = np.clip(img, a_min=0.0, a_max=1.0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, fullflname = os.path.split(img_path)
    fname, ext = os.path.splitext(fullflname)

    if is_train:
        os.makedirs(os.path.join(save_path, fname), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, fname, fname + ('' if prefix is None else f'_{prefix}.png')), img * 255.0)
    else:
        cv2.imwrite(os.path.join(save_path, fname + ('' if prefix is None else f'_{prefix}')) + ext, img * 255.0)


import matplotlib.pyplot as plt
def show(x, title="a", format="HWC", is_finish=True):
    if len(x.shape) > 3:
        print(f"Warning input image shape is {x.shape}, just show first image")
        x = x[0]
    if format == 'CHW':
        x = np.transpose(x, (1, 2, 0))
    plt.figure()
    plt.cla()
    plt.title(title)
    plt.imshow(x)
    if is_finish:
        plt.show()


def enrich_image_input(cfg, net, states):
    if cfg.img_include_states:
        # print(("states for enriching", states.shape))
        states = states[:, :, None, None] + (net[:, 0:1, :, :] * 0)
        net = torch.cat([net, states], dim=1)
    return net


# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Dict(dict):
    """
      Example:
      m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
      """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


def make_image_grid(images, per_row=2, padding=2): #  per_row =8
    npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
    assert images.shape[0] % per_row == 0
    num_rows = images.shape[0] // per_row
    image_rows = []
    for i in range(num_rows):
        image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
    return np.vstack(image_rows)


class Tee(object):

    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def write_to_file(self, data):
        self.file.write(data)

    def flush(self):
        self.file.flush()


def merge_dict(a, b):
    ret = a.copy()
    for key, val in list(b.items()):
        if key in ret:
            assert False, 'Item ' + key + 'already exists'
        else:
            ret[key] = val
    return ret


def lerp(a, b, l):
    return (1 - l) * a + l * b



class AsyncTaskManager:

    def __init__(self, target, args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.condition = threading.Condition()
        self.result = None
        self.thread = threading.Thread(target=self.worker)
        self.stopped = False
        self.thread.daemon = True
        self.thread.start()

    def worker(self):
        while True:
            self.condition.acquire()
            while self.result is not None:
                if self.stopped:
                    self.condition.release()
                    return
                self.condition.notify()
                self.condition.wait()
            self.condition.notify()
            self.condition.release()

            result = (self.target(*self.args, **self.kwargs),)

            self.condition.acquire()
            self.result = result
            self.condition.notify()
            self.condition.release()

    def get_next(self):
        self.condition.acquire()
        while self.result is None:
            self.condition.notify()
            self.condition.wait()
        result = self.result[0]
        self.result = None
        self.condition.notify()
        self.condition.release()
        return result

    def stop(self):
        while self.thread.is_alive():
            self.condition.acquire()
            self.stopped = True
            self.condition.notify()
            self.condition.release()
