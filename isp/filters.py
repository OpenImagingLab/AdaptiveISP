import numpy as np
import torch
import cv2
import math
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from isp.denoise import NonLocalMeans, NonLocalMeansGray
from isp.sharpen import unsharp_mask, adjust_sharpness, sharpness


def rgb2lum(image):
    image = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    return image[:, None, :, :]


def lerp(a, b, l):
    return (1 - l) * a + l * b


def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):
        def activation(x):
           if initial is not None:
               bias = math.atanh(2 * (initial - left) / (right - left) - 1)
           else:
               bias = 0
           return tanh01(x + bias) * (right - left) + left
        return activation
    return get_activation(l, r, initial)


class Filter(torch.nn.Module):
    def __init__(self, cfg, short_name, num_filter_parameters, predict=False):
        super(Filter, self).__init__()
        self.cfg = cfg
        self.channels = 3

        # Specified in child classes
        self.num_filter_parameters = num_filter_parameters
        self.short_name = short_name
        self.filter_parameters = None

        if predict:
            output_dim = self.get_num_filter_parameters() + self.get_num_mask_parameters()
            self.fc1 = nn.Linear(cfg.feature_extractor_dims, cfg.fc1_size)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
            # self.fc2 = nn.Linear(cfg.fc1_size, output_dim)
            self.fc_filter = nn.Linear(cfg.fc1_size, self.get_num_filter_parameters())
            self.fc_mask = nn.Linear(cfg.fc1_size, self.get_num_mask_parameters())
        self.predict = predict

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def extract_parameters(self, features):
        features = self.lrelu(self.fc1(features))
        # features = self.fc2(features)
        # print(features.shape, self.get_num_filter_parameters(),
        #       features[:, : self.get_num_filter_parameters()].shape,
        #       features[:, self.get_num_filter_parameters():].shape)
        # return features[:, : self.get_num_filter_parameters()], \
        #        features[:, self.get_num_filter_parameters():]
        return self.fc_filter(features), self.fc_mask(features)

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param):
        raise NotImplementedError("process not implement")

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    def forward(self, img, img_features=None, specified_parameter=None, high_res=None):
        if self.predict:
            assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            # filter_features.sum().backward()
            # mask_parameters.sum().backward()
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            mask_parameters = torch.zeros(1, self.get_num_mask_parameters(), dtype=torch.float32)
        if high_res is not None:
            # working on high res...
            pass
        debug_info = {}
        # We only debug the first image of this batch
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]
        self.mask_parameters = mask_parameters
        self.mask = self.get_mask(img, mask_parameters)
        debug_info['mask'] = self.mask[0]
        low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
            else:
                self.high_res_mask = self.get_mask(high_res, mask_parameters)
                high_res_output = lerp(high_res, self.process(high_res, filter_parameters), self.high_res_mask)
                high_res_output = torch.clip(high_res_output, 0.0, 1.0)
        else:
            high_res_output = None
        low_res_output = torch.clip(low_res_output, 0.0, 1.0)
        return low_res_output, high_res_output, debug_info

    def run(self, img, param):
        debug_info = {}
        # We only debug the first image of this batch
        if self.debug_info_batched():
            debug_info['filter_parameters'] = param
        else:
            debug_info['filter_parameters'] = param[0]
        self.mask = self.get_mask(img)
        debug_info['mask'] = self.mask[0]
        # print("mask", self.mask)  # tensor([[[[1.]]]])
        output = lerp(img, self.process(img, param), self.mask)
        return output

    def run_v2(self, img, param):
        debug_info = {}
        # We only debug the first image of this batch
        if self.debug_info_batched():
            debug_info['filter_parameters'] = param
        else:
            debug_info['filter_parameters'] = param[0]
        self.mask = self.get_mask(img)
        debug_info['mask'] = self.mask[0]
        # print("mask", self.mask)  # tensor([[[[1.]]]])
        output = lerp(img, self.process(img, param[None, :]), self.mask)
        return output

    def predict_param(self, img, img_features):
        filter_features, _ = self.extract_parameters(img_features)
        filter_parameters = self.filter_param_regressor(filter_features)
        self.mask = self.get_mask(img)
        output = lerp(img, self.process(img, filter_parameters), self.mask)
        return output

    def use_masking(self):
        return False

    def get_num_mask_parameters(self):
        return 6

    # Input: no need for tanh or sigmoid
    # Closer to 1 values are applied by filter more strongly
    # no additional TF variables inside
    def get_mask(self, img, mask_parameters=None):
        if not self.use_masking():
            # print('* Masking Disabled')
            return torch.ones((1, 1, 1, 1), dtype=torch.float32).to(img.device)
        # print('* Masking Enabled')
        # Six parameters for one filter
        filter_input_range = 5
        assert mask_parameters.shape[1] == self.get_num_mask_parameters()
        mask_parameters = tanh_range(l=-filter_input_range, r=filter_input_range, initial=0)(mask_parameters)
        size = list(map(int, img.shape[2:4]))
        grid = np.zeros(shape=[1] + [2] + size, dtype=np.float32)

        shorter_edge = min(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                grid[0, 0, i, j] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                grid[0, 1, i, j] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
        grid = torch.from_numpy(grid)
        # Ax + By + C * L + D
        inp = grid[:, 0, :, :, None] * mask_parameters[:, 0, None, None, None] + \
              grid[:, 1, :, :, None] * mask_parameters[:, 1, None, None, None] + \
              mask_parameters[:, 2, None, None, None] * (rgb2lum(img) - 0.5) + \
              mask_parameters[:, 3, None, None, None] * 2
        # Sharpness and inversion
        inp *= self.cfg.maximum_sharpness * mask_parameters[:, 4, None, None, None] / filter_input_range
        mask = F.sigmoid(inp)
        # Strength
        mask = mask * (mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 + 0.5) * \
               (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        # print('mask', mask.shape)
        return mask

    def visualize_filter(self, debug_info, canvas):
        # Visualize only the filter information
        assert False

    def visualize_mask(self, debug_info, res):
        return cv2.resize(debug_info['mask'].cpu().numpy() * np.ones((1, 1, 3), dtype=np.float32),
                          dsize=res, interpolation=cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(canvas, text, (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=5)
        return canvas


class ExposureFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'E', 1, predict)

    def filter_param_regressor(self, features):
        return tanh_range(-self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

    def process(self, img, param):
        return img * torch.exp(param[:, :, None, None] * np.log(2))

    def visualize_filter(self, debug_info, canvas):
        exposure = debug_info['filter_parameters'][0]
        if canvas.shape[0] == 64:
            cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
            cv2.putText(canvas, 'EV %+.2f' % exposure, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        else:
            self.draw_high_res_text('Exposure %+.2f' % exposure, canvas)


class GammaFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'G', 1, predict)

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param):
        return torch.pow(torch.clip(img, 0.001), param[:, :, None, None])

    def visualize_filter(self, debug_info, canvas):
        gamma = debug_info['filter_parameters'].detach().cpu().numpy()
        cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
        cv2.putText(canvas, 'G 1/%.2f' % (1.0 / gamma), (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class ImprovedWhiteBalanceFilter(Filter):
    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'W', 3, predict)
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor(np.array((0, 1, 1), dtype=np.float32).reshape(1, 3)).to(features.device)
        # print("ImprovedWhiteBalanceFilter", features.shape, mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling = color_scaling * (1.0 / (1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                                0.06 * color_scaling[:, 2])[:, None])
        return color_scaling

    def process(self, img, param):
        return img * param[:, :, None, None]

    def visualize_filter(self, debug_info, canvas):
        scaling = debug_info['filter_parameters'].detach().cpu().numpy()
        s = canvas.shape[0]
        cv2.rectangle(canvas, (int(s * 0.2), int(s * 0.4)), (int(s * 0.8), int(s * 0.6)),
                      list(map(float, scaling)), cv2.FILLED)


class ColorFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'C', 3 * cfg.curve_steps, predict)
        self.curve_steps = cfg.curve_steps

    def filter_param_regressor(self, features):
        # print(features.shape)
        color_curve = torch.reshape(features, shape=(-1, self.cfg.curve_steps, self.channels))[:, :, :, None, None]
        color_curve = tanh_range(*self.cfg.color_curve_range, initial=1)(color_curve)
        return color_curve

    def process(self, img, param):
        # print('img.shape, param.shape', img.shape, param.shape)
        color_curve = param  # shape [batch, 8, 3, 1, 1]
        # There will be no division by zero here unless the color filter range lower bound is 0
        color_curve_sum = torch.sum(param, dim=1) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += torch.clip(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                           color_curve[:, i, :, :, :]
        total_image *= self.cfg.curve_steps / color_curve_sum
        return total_image

    def visualize_filter(self, debug_info, canvas):
        curve = debug_info['filter_parameters'].detach().cpu().numpy()
        height, width = canvas.shape[:2]
        for i in range(self.channels):
            values = np.array([0] + list(curve[..., i, 0, 0]))
            values /= sum(values) + 1e-30
            scale = 1
            values *= scale
            for j in range(0, self.cfg.curve_steps):
                values[j + 1] += values[j]
            for j in range(self.cfg.curve_steps):
                p1 = tuple(
                        map(int, (width / self.cfg.curve_steps * j, height - 1 - values[j] * height)))
                p2 = tuple(
                        map(int, (width / self.cfg.curve_steps * (j + 1), height - 1 - values[j + 1] * height)))
                color = []
                for t in range(self.channels):
                    color.append(1 if t == i else 0)
                cv2.line(canvas, p1, p2, tuple(color), thickness=1)


class ToneFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'T', cfg.curve_steps, predict)
        self.curve_steps = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = torch.reshape(features, shape=(-1, self.cfg.curve_steps, 1))[:, :, :, None, None]
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param):
        # img = tf.minimum(img, 1.0)
        tone_curve = param  # [batch, 8, 1, 1, 1]
        tone_curve_sum = torch.sum(tone_curve, dim=1) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += torch.clip(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                           * param[:, i, :, :, :]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        img = total_image
        return img

    def visualize_filter(self, debug_info, canvas):
        curve = debug_info['filter_parameters'].detach().cpu().numpy()  # (8, 1, 1, 1)
        # print(curve.shape)
        height, width = canvas.shape[:2]
        values = np.array([0] + list(curve[..., 0, 0, 0]))
        values /= sum(values) + 1e-30
        for j in range(0, self.curve_steps):
            values[j + 1] += values[j]
        for j in range(self.curve_steps):
            p1 = tuple(
                    map(int, (width / self.curve_steps * j, height - 1 - values[j] * height)))
            p2 = tuple(
                    map(int, (width / self.curve_steps * (j + 1), height - 1 - values[j + 1] * height)))
            cv2.line(canvas, p1, p2, (0, 0, 0), thickness=1)


class ToneFilterV2(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'T', cfg.curve_steps, predict)
        self.curve_steps = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = torch.reshape(features, shape=(-1, self.cfg.curve_steps, 1))[:, :, :, None, None]
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param):
        param = param[:, :, None, None, None]
        # img = tf.minimum(img, 1.0)
        tone_curve = param  # [batch, 8, 1, 1, 1]
        tone_curve_sum = torch.sum(tone_curve, dim=1) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += torch.clip(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                           * param[:, i, :, :, :]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        img = total_image
        return img

    def visualize_filter(self, debug_info, canvas):
        curve = debug_info['filter_parameters'].detach().cpu().numpy()  # (8, 1, 1, 1)
        # print(curve.shape)
        height, width = canvas.shape[:2]
        values = np.array([0] + list(curve[..., 0, 0, 0]))
        values /= sum(values) + 1e-30
        for j in range(0, self.curve_steps):
            values[j + 1] += values[j]
        for j in range(self.curve_steps):
            p1 = tuple(
                    map(int, (width / self.curve_steps * j, height - 1 - values[j] * height)))
            p2 = tuple(
                    map(int, (width / self.curve_steps * (j + 1), height - 1 - values[j + 1] * height)))
            cv2.line(canvas, p1, p2, (0, 0, 0), thickness=1)



class ContrastFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'Ct', 1, predict)

    def filter_param_regressor(self, features):
        # return tf.sigmoid(features)
        return torch.tanh(features)

    def process(self, img, param):
        luminance = torch.clip(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])

    def visualize_filter(self, debug_info, canvas):
        exposure = debug_info['filter_parameters'][0].detach().cpu().numpy()
        cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
        cv2.putText(canvas, 'Ct %+.2f' % exposure, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class WNBFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'BW', 1, predict)

    def filter_param_regressor(self, features):
        return F.sigmoid(features)

    def process(self, img, param):
        luminance = rgb2lum(img)
        return lerp(img, luminance, param[:, :, None, None])

    def visualize_filter(self, debug_info, canvas):
        exposure = debug_info['filter_parameters'][0].detach().cpu().numpy()
        cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
        cv2.putText(canvas, 'B&W%+.2f' % exposure, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


def rgb2hsv(image):
    """
    Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
    RGB or HSV's shape: (B * C * H * W)
    RGB or HSV's range: [0, 1)
    """
    _eps = 1e-8
    hue = torch.zeros((image.shape[0], image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)
    # print(image[:, 2].dtype, image.max(1)[0].dtype, (4.0 + \
    #                                       ((image[:, 0] - image[:, 1]) / (image.max(1)[0] - image.min(1)[0] + _eps))[
    #                                           image[:, 2] == image.max(1)[0]]).dtype)
    hue[image[:, 2] == image.max(1)[0]] = 4.0 + \
                                          ((image[:, 0] - image[:, 1]) / (image.max(1)[0] - image.min(1)[0] + _eps))[
                                              image[:, 2] == image.max(1)[0]]
    hue[image[:, 1] == image.max(1)[0]] = 2.0 + \
                                          ((image[:, 2] - image[:, 0]) / (image.max(1)[0] - image.min(1)[0] + _eps))[
                                              image[:, 1] == image.max(1)[0]]
    hue[image[:, 0] == image.max(1)[0]] = (0.0 +
                                           ((image[:, 1] - image[:, 2]) / (image.max(1)[0] - image.min(1)[0] + _eps))[
                                               image[:, 0] == image.max(1)[0]]) % 6

    hue[image.min(1)[0] == image.max(1)[0]] = 0.0
    hue = hue / 6

    saturation = (image.max(1)[0] - image.min(1)[0]) / (image.max(1)[0] + _eps)
    saturation[image.max(1)[0] == 0] = 0

    value = image.max(1)[0]

    hue = hue.unsqueeze(1)
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    hsv = torch.cat([hue, saturation, value], dim=1)
    return hsv


def hsv2rgb(hsv):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    # for values out of the valid range
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    hi0 = hi == 0
    hi1 = hi == 1
    hi2 = hi == 2
    hi3 = hi == 3
    hi4 = hi == 4
    hi5 = hi == 5

    r[hi0] = v[hi0]
    g[hi0] = t[hi0]
    b[hi0] = p[hi0]

    r[hi1] = q[hi1]
    g[hi1] = v[hi1]
    b[hi1] = p[hi1]

    r[hi2] = p[hi2]
    g[hi2] = v[hi2]
    b[hi2] = t[hi2]

    r[hi3] = p[hi3]
    g[hi3] = q[hi3]
    b[hi3] = v[hi3]

    r[hi4] = t[hi4]
    g[hi4] = p[hi4]
    b[hi4] = v[hi4]

    r[hi5] = v[hi5]
    g[hi5] = p[hi5]
    b[hi5] = q[hi5]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    rgb = torch.cat([r, g, b], dim=1)
    return rgb


class SaturationPlusFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'S+', 1, predict)
        self.short_name = 'S+'
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return F.sigmoid(features)

    def process(self, img, param):
        img = torch.clip(img, min=0.0, max=1.0)
        hsv = rgb2hsv(img)
        s = hsv[:, 1:2, :, :]
        v = hsv[:, 2:3, :, :]
        # enhanced_s = s + (1 - s) * 0.7 * (0.5 - tf.abs(0.5 - v)) ** 2
        enhanced_s = s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8
        hsv1 = torch.cat([hsv[:, 0:1, :, :], enhanced_s, hsv[:, 2:, :, :]], dim=1)
        full_color = hsv2rgb(hsv1)

        param = param[:, :, None, None]
        color_param = param
        img_param = 1.0 - param

        return img * img_param + full_color * color_param

    def visualize_filter(self, debug_info, canvas):
        exposure = debug_info['filter_parameters'][0].detach().cpu().numpy()
        if canvas.shape[0] == 64:
            cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
            cv2.putText(canvas, 'S %+.2f' % exposure, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        else:
            self.draw_high_res_text('Saturation %+.2f' % exposure, canvas)


class DenoiseFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'NLM', 1, predict)
        self.num_filter_parameters = 1
        # self.denoise = NonLocalMeans(search_window_size=21, patch_size=7)
        self.denoise = NonLocalMeansGray(search_window_size=11, patch_size=5)  # gray mode 3x faster than rgb mode

    def filter_param_regressor(self, features):
        return F.sigmoid(features)
        # return tanh_range(*self.cfg.nlm_range)(features)

    def process(self, img, param):
        img = torch.clip(img, min=0.0, max=1.0)
        param = param[:, :, None, None]
        return self.denoise(img, param)

    def visualize_filter(self, debug_info, canvas):
        thresh = debug_info['filter_parameters'][0].detach().cpu().numpy()
        if canvas.shape[0] == 64:
            cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
            cv2.putText(canvas, 'NLM %+.2f' % thresh, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        else:
            self.draw_high_res_text('NLM %+.2f' % thresh, canvas)


class SharpenUSMFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'USM', 2, predict)
        self.num_filter_parameters = 2

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_sharpen_range)(features)

    def process(self, img, param):
        # param shape [batch, 2]
        return unsharp_mask(img, param[:, 0], param[:, 1], kernel_size=(5, 5), clip=True)

    def visualize_filter(self, debug_info, canvas):
        sigma = debug_info['filter_parameters'][0].detach().cpu().numpy()
        amount = debug_info['filter_parameters'][1].detach().cpu().numpy()
        if canvas.shape[0] == 64:
            cv2.putText(canvas, 'Sharpen', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 's %+.2f' % (sigma, ), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 'a %+.2f' % (amount, ), (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
        else:
            self.draw_high_res_text('Sharpen %+.2f %+.2f' % (sigma, amount), canvas)


class SharpenFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'Shr', 1, predict)

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.sharpen_range)(features)

    def process(self, img, param):
        # param shape [batch, 1]
        return adjust_sharpness(img, param[:, :, None, None])

    def visualize_filter(self, debug_info, canvas):
        sigma = 5
        amount = debug_info['filter_parameters'].detach().cpu().numpy()
        if canvas.shape[0] == 64:
            cv2.putText(canvas, 'Sharpen', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 's %+.2f' % (sigma, ), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 'a %+.2f' % (amount, ), (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
        else:
            self.draw_high_res_text('Sharpen %+.2f %+.2f' % (sigma, amount), canvas)


class SharpenFilterV2(Filter):
    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'Shr', 1, predict)

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.sharpen_range)(features)

    def process(self, img, param):
        # param shape [batch, 1]
        return sharpness(img, param[:, :, None, None])

    def visualize_filter(self, debug_info, canvas):
        sigma = 5
        amount = debug_info['filter_parameters'].detach().cpu().numpy()
        if canvas.shape[0] == 64:
            cv2.putText(canvas, 'Sharpen', (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 's %+.2f' % (sigma, ), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
            cv2.putText(canvas, 'a %+.2f' % (amount, ), (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
        else:
            self.draw_high_res_text('Sharpen %+.2f %+.2f' % (sigma, amount), canvas)


def color_correction_matrix(image, ccm):
    """ image: NCHW; ccm: N,3,3; return: NCHW """
    image = torch.permute(image, (0, 2, 3, 1))  # NCHW -> NHWC
    image = image[:, :, :, None, :]  # NHW1C
    ccm = ccm[:, None, None, :, :]   # N33 -> N1133
    out = torch.sum(image * ccm, dim=-1)
    return torch.permute(out, (0, 3, 1, 2))


def test_color_correction_matrix():
    aa = np.random.rand(8, 8, 3)
    ccm = np.array([[0.5, 0.2, 0.6], [0.5, 0.2, 0.8], [0.5, 0.2, 0.9]])
    image = np.reshape(aa, [-1, 3])
    image = np.tensordot(image, ccm, [[-1], [-1]])
    np_res = image.reshape(aa.shape)
    print(np_res.shape)  # (8, 8, 3)

    a = torch.from_numpy(np.transpose(np.expand_dims(aa, 0), (0, 3, 1, 2)))
    b = torch.from_numpy(ccm)
    print("a", a.shape)
    # a = torch.permute(a, (0, 2, 3, 1))
    out = color_correction_matrix(a, b[None, :, :])
    print(out.shape)
    print(out)
    print(image)
    print(out.squeeze(0).numpy().transpose((1, 2, 0)) == np_res)


class CCMFilter(Filter):

    def __init__(self, cfg, predict=False):
        Filter.__init__(self, cfg, 'CCM', 9, predict)

    def filter_param_regressor(self, features):
        ccm = tanh_range(*self.cfg.ccm_range)(features)
        return ccm

    def process(self, img, param):
        # param  [batch, 9]
        # img [batch, 3, H, W]
        param = torch.reshape(param, shape=(-1, 3, 3))
        param = param / torch.sum(param, dim=-1, keepdim=True)
        return color_correction_matrix(img, param)

    def visualize_filter(self, debug_info, canvas):
        ccm = debug_info['filter_parameters'].detach().cpu().numpy()  # [9]
        ccm = np.reshape(ccm, (3, 3))
        ccm = ccm / np.sum(ccm, axis=-1, keepdims=True)
        if canvas.shape[0] == 64:
            cv2.putText(canvas, '%+.2f %+.2f %+.2f' % (ccm[0][0], ccm[0][1], ccm[0][2]), (1, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0))
            cv2.putText(canvas, '%+.2f %+.2f %+.2f' % (ccm[1][0], ccm[1][1], ccm[1][2]), (1, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0))
            cv2.putText(canvas, '%+.2f %+.2f %+.2f' % (ccm[2][0], ccm[2][1], ccm[2][2]), (1, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0))
        else:
            self.draw_high_res_text('CCM %+.2f %+.2f %+.2f; %+.2f %+.2f %+.2f; %+.2f %+.2f %+.2f' % (
                ccm[0][0], ccm[0][1], ccm[0][2], ccm[1][0], ccm[1][1], ccm[1][2], ccm[2][0], ccm[2][1], ccm[2][2]), canvas)


if __name__ == "__main__":
    from PIL import Image
    from util import show
    import rawpy
    from unprocess_np import unprocess_wo_mosaic

    filename = '../images/000000581781.jpg'
    image = Image.open(filename)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    W, H = image.size
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.
    rgb, metadata = unprocess_wo_mosaic(image)

    import rawpy
    raw_file = "../images/LRM_20191230_052456_1.dng"
    raw = rawpy.imread(raw_file)
    rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB,
                          gamma=(1, 1), no_auto_bright=True, output_bps=8, use_camera_wb=False, use_auto_wb=False,
                          user_wb=(1, 1, 1, 1))
    rgb = np.array(rgb).astype(np.float32) / 255.

    # show(rgb, "rgb")
    rgb_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))[None, :, :, :]
    cfg = EasyDict({"fc1_size": 4096, "curve_steps": 8, "feature_extractor_dims": 4096})

    exp = ExposureFilter(cfg)
    param = torch.tensor([0.09012079], dtype=torch.float32)[None, :]
    # res = exp.process(rgb_t, param)
    res = exp.run(rgb_t, param)
    show(res[0].numpy(), "exp", format='CHW', is_finish=False)

    wb = ImprovedWhiteBalanceFilter(cfg)
    param = torch.tensor([2.4052505, 1.2233436, 1.8800205], dtype=torch.float32)[None, :]
    # res = wb.process(res, param)
    res = wb.run(res, param)
    show(res[0].numpy(), "wb", format='CHW', is_finish=False)

    gamma = GammaFilter(cfg)
    param = torch.tensor([0.38566995], dtype=torch.float32)[None, :]
    # res = gamma.process(res, param)
    res = gamma.run(res, param)
    show(res[0].numpy(), "gamma", format='CHW', is_finish=False)

    color = ColorFilter(cfg)
    a = np.array([[0.32030997, 0.9885651, 1.0990355, 1.325188, 1.2513527, 1.122059, 0.8997669, 0.40487906],
     [0.31780338, 1.2305514, 1.140491,   1.6768382,  1.5306898,  1.1610103, 1.0416414, 0.9278361],
     [0.42245385, 0.82483983, 1.0253056, 1.3527378, 1.1259661, 1.0415019, 0.8615291, 0.55761546]])
    param = torch.tensor(a.transpose((1, 0)), dtype=torch.float32)[None, :, :, None, None, ]
    # res = color.process(res, param)
    res = color.run(res, param)
    show(res[0].numpy(), "ColorFilter", format='CHW', is_finish=False)

    tone = ToneFilter(cfg)
    param = torch.tensor([0.33231276, 0.57279795, 0.7360831, 0.74742377, 1.0396448, 1.0650746, 1.1307411, 0.67496914],
                         dtype=torch.float32)[None, :, None, None, None]
    # res = color.process(res, param)
    res = tone.run(res, param)
    show(res[0].numpy(), "ToneFilter", format='CHW', is_finish=False)

    contrast = ContrastFilter(cfg)
    param = torch.tensor([-0.38305384], dtype=torch.float32)[None, :]
    # res = contrast.process(res, param)
    res = contrast.run(res, param)
    show(res[0].numpy(), "ContrastFilter", format='CHW', is_finish=False)

    saturation = SaturationPlusFilter(cfg)
    param = torch.tensor([-0.06362316], dtype=torch.float32)[None, :]
    # res = saturation.process(res, param)
    res = saturation.run(res, param)
    show(res[0].numpy(), "SaturationPlusFilter", format='CHW', is_finish=False)

    wnb = WNBFilter(cfg)
    param = torch.tensor([-0.2960269], dtype=torch.float32)[None, :]
    # res = wnb.process(res, param)
    res = wnb.run(res, param)
    show(res[0].numpy(), "WNBFilter", format='CHW', is_finish=False)

    denoise = DenoiseFilter(cfg).cuda()
    param = torch.tensor([0.6], dtype=torch.float32)[None, :].cuda()
    # res = exp.process(res, param)
    res = denoise.run(res.cuda(), param).cpu()
    show(res[0].numpy(), "Denoise", format='CHW', is_finish=False)

    sharpen = SharpenFilter(cfg)
    param = torch.tensor([0.5], dtype=torch.float32)[None, :]
    # res = exp.process(res, param)
    res = sharpen.run(res, param)
    show(res[0].numpy(), "Sharpen", format='CHW', is_finish=True)
