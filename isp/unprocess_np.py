# -*- coding: utf-8 -*-
import numpy as np


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = np.array(xyz2cams)
    weights = np.random.uniform(1e-8, 1e8, size=(num_ccms, 1, 1))
    #  weights = np.ones((num_ccms, 1, 1))
    weights_sum = np.sum(weights, axis=0)
    xyz2cam = np.sum(xyz2cams * weights, axis=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
    return rgb2cam

def get_calibrated_cam2rgb():
    cam2rgb_matrix = np.array([[ 2.04840695, -1.27161572,  0.22320878],
                               [-0.22163155,  1.77694640, -0.55531485],
                               [-0.00770995, -0.59257895,  1.60028890]], dtype=np.float32)
    return cam2rgb_matrix

def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / np.random.normal(0.8, 0.1)

    # Red and blue gains represent white balance.
    red_gain = np.random.uniform(1.9, 2.4)
    blue_gain = np.random.uniform(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = np.clip(image, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)

def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    return np.maximum(image, 1e-8) ** 2.2

def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.shape
    image = np.reshape(image, [-1, 3])
    image = np.tensordot(image, ccm, [[-1], [-1]])
    return np.reshape(image, shape)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    gains = np.stack((1.0 / red_gain, 1.0, 1.0 / blue_gain)) / rgb_gain
    gains = gains.squeeze()
    gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = np.mean(image, axis=-1, keepdims=True)
    inflection = 0.9
    mask = (np.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = np.maximum(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains

def mosaic(image, pattern="RGGB"):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    red = image[0::2, 0::2, 0]
    green_red = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    if pattern.upper() == "RGGB":
        out = np.stack((red, green_red, green_blue, blue), axis=-1) # RGGB
    elif pattern.upper() == "RGBG":
        out = np.stack((red, green_red, blue, green_blue), axis=-1) # RGBG  Cannon 5D Mark IV
    else:
        raise ValueError(f"Don't implement pattern: {pattern.upper()}")

    out = np.reshape(out, (shape[0] // 2, shape[1] // 2, 4)) # 4 channel
    # out = reconstruct_bayer(out, pattern) # 1 channel
    return out

def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1)),
            'rgbg': ((0, 0), (1, 0), (1, 1), (0, 1))}[pattern.lower()]

def reconstruct_bayer(raw, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param raw: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg' | 'rgbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)
    sub_arrays = [raw[::, ::, 0], raw[::, ::, 1], raw[::, ::, 2], raw[::, ::, 3]]
    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array


def adjust_random_brightness(image, s_range=(0.1, 0.3)):
    if isinstance(s_range, (list, tuple)):
        assert s_range[0] < s_range[1], "s_range[0] should less than s_range[1]"
        ratio = np.random.rand() * (s_range[1] - s_range[0]) + s_range[0]
    else:
        ratio = s_range
    return image * ratio, ratio


def add_gaussian_noise(image, mean=0, std=0.25):
    noise = np.random.normal(mean, std, image.shape)
    return image + noise


def random_noise_levels_log(shot_noise=None):
    """Generates random noise levels from a log-log linear distribution."""
    if shot_noise is None:
        log_min_shot_noise = np.log(0.0001)
        log_max_shot_noise = np.log(0.012)
        log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
        shot_noise = np.exp(log_shot_noise)
    else:
        log_shot_noise = np.log(shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise


def random_noise_levels_linear(shot_noise=None):
    """Generates random noise levels from a linear distribution."""
    if shot_noise is None:
        min_shot_noise = 0.0001
        max_shot_noise = 0.012
        shot_noise = np.random.uniform(min_shot_noise, max_shot_noise)
        log_shot_noise = np.log(shot_noise)
    else:
        log_shot_noise = np.log(shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise


def add_read_and_shot_noise(image, shot_noise=0.01, read_noise=0.005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = np.random.normal(0, np.sqrt(variance), size=variance.shape)
    return image + noise


def unprocess_canon(image):
    """Unprocesses an image from sRGB to realistic raw data."""

    # Randomly creates image metadata.
    # rgb2cam = random_ccm()
    # cam2rgb = np.linalg.inv(rgb2cam)
    cam2rgb = get_calibrated_cam2rgb()
    rgb2cam = np.linalg.inv(cam2rgb)
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image, "RGBG")

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'cfa': "RGBG",
    }
    return image, metadata


def unprocess(image, pattern="RGGB"):
    """Unprocesses an image from sRGB to realistic raw data."""

    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image, pattern)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'cfa': "RGGB",
    }
    return image, metadata


def unprocess_wo_mosaic(image, add_noise=False, brightness_range=None, noise_level=None, use_linear=False):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    image, _ = adjust_random_brightness(image, s_range=0.9)  # (0.6, 0.8)

    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    # image = mosaic(image, pattern)

    gain = 1.0
    if brightness_range is not None:
        image, gain = adjust_random_brightness(image, s_range=brightness_range)

    shot, read = 0.0, 0.0
    if add_noise:
        if use_linear:
            shot, read = random_noise_levels_linear(noise_level)
        else:
            shot, read = random_noise_levels_log(noise_level)
        image = add_read_and_shot_noise(image, shot, read)
        image = np.clip(image, 0.0, 1.0)
    
    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'cfa': "RGGB",
        'gain': gain,
        'noise': (shot, read),
    }
    return image, metadata


def unprocess_wo_mosaic_v2(image, add_noise=False, brightness_range=None, noise_level=None, use_linear=False):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    # image, _ = adjust_random_brightness(image, s_range=0.9)  # (0.6, 0.8)

    # # Approximately inverts global tone mapping.
    # image = inverse_smoothstep(image)
    # # Inverts gamma compression.
    # image = gamma_expansion(image)
    # # Inverts color correction.
    # image = apply_ccm(image, rgb2cam)
    # # Approximately inverts white balance and brightening.
    # image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)

    image, _ = adjust_random_brightness(image, s_range=(0.5, 0.9))  # (0.6, 0.8)
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    image = gamma_expansion(image)
    image = inverse_smoothstep(image)
    image = apply_ccm(image, rgb2cam)

    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    # image = mosaic(image, pattern)

    gain = 1.0
    if brightness_range is not None:
        image, gain = adjust_random_brightness(image, s_range=brightness_range)

    shot, read = 0.0, 0.0
    if add_noise:
        if use_linear:
            shot, read = random_noise_levels_linear(noise_level)
        else:
            shot, read = random_noise_levels_log(noise_level)
        image = add_read_and_shot_noise(image, shot, read)
        image = np.clip(image, 0.0, 1.0)

    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'cfa': "RGGB",
        'gain': gain,
        'noise': (shot, read),
    }
    return image, metadata


def unprocess_wo_mosaic_inter(image, add_noise=False, brightness_range=None, noise_level=None, use_linear=False):
    """Unprocesses an image from sRGB to realistic raw data."""

    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = np.linalg.inv(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()

    image, _ = adjust_random_brightness(image, s_range=0.9)  # (0.6, 0.8)

    inter_res = {}
    inter_res['rgb'] = image
    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    inter_res['tone_mapping'] = image
    # Inverts gamma compression.
    image = gamma_expansion(image)
    inter_res['gamma'] = image
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    inter_res['ccm'] = image
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    inter_res['gain'] = image
    # Clips saturated pixels.
    image = np.clip(image, 0.0, 1.0)

    # Applies a Bayer mosaic.
    # image = mosaic(image, pattern)
    gain = 1.0
    if brightness_range is not None:
        image, gain = adjust_random_brightness(image, s_range=brightness_range)
    inter_res['brightness'] = image
    shot, read = 0.0, 0.0
    if add_noise:
        if use_linear:
            shot, read = random_noise_levels_linear(noise_level)
        else:
            shot, read = random_noise_levels_log(noise_level)
        image = add_read_and_shot_noise(image, shot, read)
        image = np.clip(image, 0.0, 1.0)
    inter_res['noisy'] = image
    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
        'cfa': "RGGB",
        'gain': gain,
        'noise': (shot, read),
    }
    return image, metadata, inter_res



import matplotlib.pyplot as plt
def show(x, title="a", format="HWC", is_finish=True):
    if format == 'CHW':
        x = np.transpose(x, (1, 2, 0))
    plt.figure()
    plt.cla()
    plt.title(title)
    plt.imshow(x)
    if is_finish:
        plt.show()


if __name__ == "__main__":
    # import numpy as np
    # import cv2
    #
    # filename = 'images/000000270705.jpg'
    # img = cv2.imread(filename)
    # # Displaying the image
    # # cv2.imshow("RGB", img)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    #
    # print(np.max(img), np.min(img))
    # # show(img, is_finish=False)
    # # show(np.clip(img/255, 0, 1.0))
    # # img = np.clip(img / 255, 0, 0.9)
    # # img = adjust_random_brightness(img / 255)
    # # img = inverse_smoothstep(img)
    #
    # cv2.imshow("RGB", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    import cv2
    import rawpy
    from PIL import Image
    np.random.seed(0)
    filename = 'images/000000270705.jpg'
    # filename = 'output/000000581781.jpg'
    image = Image.open(filename)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    W, H = image.size
    image = image.resize((W // 2 * 2, H // 2 * 2))
    image = np.array(image).astype(np.float32) / 255.

    rgb, metadata, inter_res = unprocess_wo_mosaic_inter(image, True)
    print(metadata)
    # {'cam2rgb': array([[ 1.64968762, -0.58507444, -0.06461318],
    #        [-0.18113312,  1.4904194 , -0.30928628],
    #        [ 0.02480729, -0.42786895,  1.40306166]]),
    #        'rgb_gain': 1.013421964176166, 'red_gain': 2.118793605631346, 'blue_gain': 1.8567092003128318, 'cfa': 'RGGB'}
    show(inter_res['rgb'], "rgb", is_finish=False)
    show(inter_res['tone_mapping'], "tone_mapping", is_finish=False)
    show(inter_res['gamma'], "gamma", is_finish=False)
    show(inter_res['ccm'], "ccm", is_finish=False)
    show(inter_res['gain'], "gain", is_finish=False)
    show(inter_res['brightness'], "brightness", is_finish=False)
    show(inter_res['noisy'], "noisy", is_finish=False)
    show(rgb, "raw", is_finish=True)

    # for k, v in inter_res.items():
    #     img = Image.fromarray(np.array(v*255, dtype=np.uint8), 'RGB')
    #     img.save(f"output/unprocess_adjust_bright_results/000000270705_{k}.png")

