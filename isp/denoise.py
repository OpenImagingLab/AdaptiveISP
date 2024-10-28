# https://ridiqulous.com/pytorch-non-local-means/

import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-8


def rgb_to_luminance(rgb_tensor):
    # :param rgb_tensor: torch.Tensor(N, 3, H, W, ...) in [0, 1] range
    # :return: torch.Tensor(N, 1, H, W, ...) in [0, 1] range
    rgb_tensor = torch.clip(rgb_tensor, 0.0, 1.0)
    assert rgb_tensor.min() >= 0.0 and rgb_tensor.max() <= 1.0, \
        f"DenoiseFilter rgb_tensor is {rgb_tensor.min()}, {rgb_tensor.max()}"
    return 0.299 * rgb_tensor[:, :1, ...] + 0.587 * rgb_tensor[:, 1:2, ...] + 0.114 * rgb_tensor[:, 2:, ...]


class ShiftStack(nn.Module):
    """
    Shift n-dim tensor in a local window and generate a stacked
    (n+1)-dim tensor with shape (*orig_shapes, w*y), where wx
    and wy are width and height of the window
    """

    def __init__(self, window_size):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ..., w*y)
        shifted_tensors = []
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                shifted_tensors.append(
                    torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))
                )
        return torch.stack(shifted_tensors, dim=-1)


class BoxFilter(nn.Module):
    def __init__(self, window_size, reduction='mean'):
        # :param window_size: Int or Tuple(Int, Int) in (win_width, win_height) order
        # :param reduction: 'mean' | 'sum'
        super().__init__()
        wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
        assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
        self.rx, self.ry = wx // 2, wy // 2
        self.area = wx * wy
        self.reduction = reduction

    def forward(self, tensor):
        # :param tensor: torch.Tensor(N, C, H, W, ...)
        # :return: torch.Tensor(N, C, H, W, ...)
        local_sum = torch.zeros_like(tensor)
        for x_shift in range(-self.rx, self.rx + 1):
            for y_shift in range(-self.ry, self.ry + 1):
                local_sum += torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))

        return local_sum if self.reduction == 'sum' else local_sum / self.area


class NonLocalMeans(nn.Module):
    def __init__(self, search_window_size=21, patch_size=7):
        super().__init__()
        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        self.r = search_window_size // 2

    def forward(self, rgb, h):
        batch_size, _, height, width = rgb.shape
        weights = torch.zeros((batch_size, 3, height, width)).float().to(rgb.device)  # (N, 3, H, W)
        denoised_rgb = torch.zeros_like(rgb)  # (N, 3, H, W)

        for x_shift in range(-self.r, self.r + 1):
            for y_shift in range(-self.r, self.r + 1):
                shifted_rgb = torch.roll(rgb, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 3, H, W)
                # distance = torch.sqrt(self.box_sum((rgb - shifted_rgb) ** 2) + EPS)  # (N, 3, H, W)
                # weight = torch.exp(-distance / (torch.pow(h, 2) + EPS))  # (N, 3, H, W)
                distance = torch.sqrt(torch.relu(self.box_sum((rgb - shifted_rgb) ** 2)))  # (N, 1, H, W)
                weight = torch.exp(-distance / (torch.relu(h) + EPS))  # (N, 1, H, W)

                denoised_rgb += shifted_rgb * weight  # (N, 3, H, W)
                weights += weight  # (N, 3, H, W)

        return torch.clamp(denoised_rgb / weights, 0., 1.)  # (N, 3, H, W)


class NonLocalMeansGray(nn.Module):
    def __init__(self, search_window_size=21, patch_size=7):
        super().__init__()
        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        self.r = search_window_size // 2

    def forward(self, rgb, h):
        batch_size, _, height, width = rgb.shape
        weights = torch.zeros((batch_size, 1, height, width)).float().to(rgb.device)  # (N, 1, H, W)
        denoised_rgb = torch.zeros_like(rgb)  # (N, 3, H, W)

        y = rgb_to_luminance(rgb)  # (N, 1, H, W)

        for x_shift in range(-self.r, self.r + 1):
            for y_shift in range(-self.r, self.r + 1):
                shifted_rgb = torch.roll(rgb, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 3, H, W)
                shifted_y = torch.roll(y, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 1, H, W)

                # distance = torch.sqrt(self.box_sum((y - shifted_y) ** 2) + EPS)  # (N, 1, H, W)
                # weight = torch.exp(-distance / (torch.pow(h, 2) + EPS))  # (N, 1, H, W)
                distance = torch.sqrt(torch.relu(self.box_sum((y - shifted_y) ** 2)))  # (N, 1, H, W)
                weight = torch.exp(-distance / (torch.relu(h) + EPS))  # (N, 1, H, W)

                denoised_rgb += shifted_rgb * weight  # (N, 3, H, W)
                weights += weight  # (N, 1, H, W)

        return torch.clamp(denoised_rgb / weights, 0., 1.)  # (N, 3, H, W)


class NonLocalMeansParam(nn.Module):
    def __init__(self, h0, search_window_size=21, patch_size=7):
        super().__init__()
        self.h = nn.Parameter(torch.tensor([float(h0)]), requires_grad=True)

        self.box_sum = BoxFilter(window_size=patch_size, reduction='sum')
        self.r = search_window_size // 2
        self.gen_window_stack = ShiftStack(window_size=search_window_size)
        self.search_window_size = search_window_size

    def forward(self, rgb):

        # ---- unfold -----
        y = rgb_to_luminance(rgb)  # (N, 1, H, W)
        pad = (self.search_window_size - 1) // 2
        rgb_pad = F.pad(rgb, pad=[pad, pad, pad, pad], mode='reflect')
        rgb_window_stack = rgb_pad.unfold(2, self.search_window_size, 1).unfold(3, self.search_window_size, 1) # (N, 3, H, W, w*y)
        rgb_window_stack = rgb_window_stack.reshape((rgb_window_stack.shape[0], rgb_window_stack.shape[1], rgb_window_stack.shape[2], rgb_window_stack.shape[3], -1))
        #  = self.gen_window_stack(y)  # (N, 1, H, W, w*y)
        y_pad = F.pad(y, pad=[pad, pad, pad, pad], mode='reflect')
        y_window_stack = y_pad.unfold(2, self.search_window_size, 1).unfold(3, self.search_window_size, 1)  # (N, 1, H, W, w*y)
        y_window_stack = y_window_stack.reshape((y_window_stack.shape[0], y_window_stack.shape[1],
                                                 y_window_stack.shape[2], y_window_stack.shape[3], -1))
        # print(rgb_window_stack.shape, y_window_stack.shape)

        dis = (y.unsqueeze(-1) - y_window_stack) ** 2
        # print(dis.shape)
        dis = F.pad(dis, pad=[0, 0, pad, pad, pad, pad], mode='reflect')
        dis = dis.unfold(2, self.search_window_size, 1).unfold(3, self.search_window_size, 1)
        dis = torch.sum(dis, dim=(-1, -2))

        distances = torch.sqrt(torch.relu(dis))
        weights = torch.exp(-distances / (torch.relu(self.h) + EPS))  # (N, 1, H, W, w*y)

        denoised_rgb = (weights * rgb_window_stack).sum(dim=-1) / weights.sum(dim=-1)  # (N, 3, H, W)
        return torch.clamp(denoised_rgb, 0, 1)  # (N, 3, H, W)

        # batch_size, _, height, width = rgb.shape
        # weights = torch.zeros((batch_size, 1, height, width)).float().to(rgb.device)  # (N, 1, H, W)
        # denoised_rgb = torch.zeros_like(rgb)  # (N, 3, H, W)
        # y = rgb_to_luminance(rgb)  # (N, 1, H, W)
        # shifted_rgb_list = []
        # weights_list = []
        # for x_shift in range(-self.r, self.r + 1):
        #     for y_shift in range(-self.r, self.r + 1):
        #         shifted_rgb = torch.roll(rgb, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 3, H, W)
        #         shifted_y = torch.roll(y, shifts=(y_shift, x_shift), dims=(2, 3))  # (N, 1, H, W)
        #
        #         # distance = torch.sqrt(self.box_sum((y - shifted_y) ** 2) + EPS)  # (N, 1, H, W)
        #         # weight = torch.exp(-distance / (torch.pow(self.h, 2) + EPS))  # (N, 1, H, W)
        #         distance = torch.sqrt(torch.relu(self.box_sum((y - shifted_y) ** 2)))  # (N, 1, H, W)
        #         weight = torch.exp(-distance / (torch.relu(h) + EPS))  # (N, 1, H, W)
        #
        #         shifted_rgb_list.append(shifted_rgb)
        #         weights_list.append(weight)
        #
        # weights_list_sum = torch.stack(weights_list, dim=1).sum(dim=1)
        # for i in range(len(shifted_rgb_list)):
        #     denoised_rgb += shifted_rgb_list[i] * weights_list[i] / weights_list_sum
        # return torch.clamp(denoised_rgb, 0, 1)  # (N, 3, H, W)


def gen_image_pair(image_path, device, sigma):

    # :return: two tensors with shape (1, 3, H, W) in [0, 1] range
    clean_rgb = skimage.io.imread(image_path).astype(np.float32) / 255.0
    import cv2
    clean_rgb = cv2.resize(clean_rgb, (64, 64))
    # additive white gaussian noise
    awgn = np.random.normal(0, scale=sigma, size=clean_rgb.shape).astype(np.float32)
    noisy_rgb = np.clip(clean_rgb + awgn, 0, 1)

    clean_rgb = torch.from_numpy(clean_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    noisy_rgb = torch.from_numpy(noisy_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return clean_rgb, noisy_rgb


def test_mem():
    dev = torch.device('cuda')
    clean_rgb, noisy_rgb = gen_image_pair('images/000000581781.jpg', device=dev, sigma=0.05)

    denoiser = NonLocalMeansParam(h0=0.05).to(dev)
    denoiser.train()
    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=0.0001, weight_decay=0.0001)
    loss = nn.MSELoss()

    for iter_num in range(1000):
        noisy_rgb = torch.tensor(noisy_rgb, requires_grad=True)
        denoised_rgb = denoiser(noisy_rgb)
        mse = loss(clean_rgb, denoised_rgb)

        mse.backward()
        print(noisy_rgb.grad)
        optimizer.step()

        psnr = 10 * torch.log10(1.0 / mse)
        print('step{}: PSNR={:.3f} (h={:.5f})'.format(iter_num, psnr.item(), float(denoiser.h.item())))
        break


def test_cv2_nlm():
    import numpy as np
    # importing the openCV library
    import cv2
    # importing the pyplot for visualizing the image
    from matplotlib import pyplot as plt

    # reading image using the cv2.imread() function
    img = cv2.imread('images/LRM_20191230_052456_1.dng')

    # denoising the image using the cv2.fastNlMeansDenoising() function
    dst = cv2.fastNlMeansDenoising(img, None, 15, 7, 21)

    # visualizing the image and comparing noisy image and image after denoising
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()


if __name__ == "__main__":
    # test_cv2_nlm()
    # test_mem()
    # exit()


    from PIL import Image
    import cv2
    dev = torch.device('cuda')
    # filename = 'LOD/RAW_Dark/2.png'
    # image = Image.open(filename)
    # if image.mode != 'RGB':
    #     image = image.convert("RGB")
    # W, H = image.size
    # image = image.resize((W // 2 * 2, H // 2 * 2))
    # image = np.array(image).astype(np.float32) / 255.
    # noisy_rgb = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(dev)

    import rawpy
    raw = rawpy.imread("images/LRM_20191230_052456_1.dng")
    # raw.raw_pattern[:] = np.array([[0, 1], [2, 1]], np.uint8)
    # raw.color_desc = b"RGGB"
    image = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB,
                            gamma=(1, 1), no_auto_bright=True, output_bps=8, use_camera_wb=False, use_auto_wb=False,
                            user_wb=(1, 1, 1, 1))
    image = np.array(image).astype(np.float32) / 255.
    image = cv2.resize(image, (1080, 720))
    noisy_rgb = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    noisy_rgb = torch.cat([noisy_rgb, noisy_rgb], dim=0)

    # clean_rgb, noisy_rgb = gen_image_pair('images/000000581781.jpg', device=dev, sigma=50/255)
    denoiser = NonLocalMeansGray(search_window_size=11, patch_size=5).to(dev)
    # h = torch.tensor([[0.0], [1.0]]).to(dev)
    h = torch.tensor([[1.0], [0.0]]).to(dev)
    noisy_rgb = torch.tensor(noisy_rgb, requires_grad=True)
    denoised_rgb = denoiser(noisy_rgb, h[:, :, None, None])
    from util import show
    # show(clean_rgb.detach().cpu().numpy(), title="clean", format="CHW", is_finish=False)
    show(noisy_rgb.detach().cpu().numpy(), title="noisy", format="CHW", is_finish=False)
    show(denoised_rgb.detach().cpu().numpy()[0], title="denoise", format="CHW", is_finish=False)
    show(denoised_rgb.detach().cpu().numpy()[1], title="denoise", format="CHW", is_finish=True)
    denoised_rgb.mean().backward()
    print(noisy_rgb.grad)
