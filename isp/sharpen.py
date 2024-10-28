import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import conv2d
import numpy as np
import math
from typing import List, Tuple
from torch import Tensor

import torchvision
from torchvision.transforms.functional_tensor import torch_pad # _cast_squeeze_in, _cast_squeeze_out, 
# from torchvision.transforms.functional_tensor import _max_value


def _get_gaussian_kernel1d(kernel_size: int, sigma: Tensor) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: Tensor, dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _cast_squeeze_in(img: Tensor, req_dtypes: List[torch.dtype]) -> Tuple[Tensor, bool, bool, torch.dtype]:
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype) -> Tensor:
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def gaussian_blur_torch(img, kernel_size: List[int], sigma: Tensor) -> Tensor:
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        img,
        [
            kernel.dtype,
        ],
    )

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


def unsharp_mask(img, sigma, amount, kernel_size=(5, 5), clip=True):
    # x: N,C,H,W
    # sigma: 0-2
    # amount: 0-2
    # blured = torchvision.transforms.functional.gaussian_blur(img, (5, 5), sigma=sigma)
    # if every batch image use different sigma and amount, run it with batch by batch
    # print(img.shape, sigma.shape, amount.shape)
    if len(img.shape) > 3 and sigma.shape[0] > 1:
        assert img.shape[0] == sigma.shape[0] == amount.shape[0], "input batch shape error"
        out = torch.zeros_like(img)
        for b in range(img.shape[0]):
            blured = gaussian_blur_torch(img[b], kernel_size=kernel_size, sigma=sigma[b])
            out[b] = img[b] + (img[b] - blured) * amount[b]
    else:
        blured = gaussian_blur_torch(img, kernel_size=kernel_size, sigma=sigma.squeeze(-1))
        out = img + (img - blured) * amount
    if clip:
        out = torch.clip(out, 0.0, 1.0)
    return out


def adjust_sharpness(image: torch.Tensor, factor: (torch.Tensor, float)) -> torch.Tensor:
    num_channels = image.shape[1]

    # The following is a normalized 3x3 kernel with 1s in the edges and a 5 in the middle.
    # kernel_dtype = image.dtype
    # a, b = 1.0 / 13.0, 5.0 / 13.0
    # kernel = torch.tensor([[a, a, a], [a, b, a], [a, a, a]], dtype=kernel_dtype, device=image.device)
    # kernel = kernel.expand(num_channels, 1, 3, 3)
    #
    # kernel_size = [3, 3]
    # padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    # # image = torch_pad(image, padding, mode="reflect")
    # blurred = conv2d(image, kernel, groups=num_channels)

    kernel = torch.ones((3, 3), dtype=image.dtype, device=image.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(image.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        image,
        [
            kernel.dtype,
        ],
    )
    result_tmp = conv2d(result_tmp, kernel, groups=num_channels)
    blurred = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)
    # blurred = conv2d(image, kernel, groups=num_channels)
    mask = torch.ones_like(blurred)
    kernel_size = [3, 3]
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    padded_mask = torch.nn.functional.pad(mask, padding)
    padded_blurred = torch.nn.functional.pad(blurred, padding)
    blurred = torch.where(padded_mask == 1, padded_blurred, image)

    output = image * factor + blurred * (1.0 - factor)
    output = torch.clip(output, 0.0, 1.0)
    return output


def sharpness(image: torch.Tensor, factor: (torch.Tensor, float)) -> torch.Tensor:
    num_channels = image.shape[1]

    # The following is a normalized 3x3 kernel with 1s in the edges and a 5 in the middle.
    # kernel_dtype = image.dtype
    # a, b = 1.0 / 13.0, 5.0 / 13.0
    # kernel = torch.tensor([[a, a, a], [a, b, a], [a, a, a]], dtype=kernel_dtype, device=image.device)
    # kernel = kernel.expand(num_channels, 1, 3, 3)
    #
    # kernel_size = [3, 3]
    # padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    # # image = torch_pad(image, padding, mode="reflect")
    # blurred = conv2d(image, kernel, groups=num_channels)

    kernel = torch.ones((3, 3), dtype=image.dtype, device=image.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(image.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(
        image,
        [
            kernel.dtype,
        ],
    )
    result_tmp = conv2d(result_tmp, kernel, groups=num_channels)
    blurred = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)
    # blurred = conv2d(image, kernel, groups=num_channels)
    mask = torch.ones_like(blurred)
    kernel_size = [3, 3]
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    padded_mask = torch.nn.functional.pad(mask, padding)
    padded_blurred = torch.nn.functional.pad(blurred, padding)
    blurred = torch.where(padded_mask == 1, padded_blurred, image)

    output = image + (image - blurred) * factor
    output = torch.clip(output, 0.0, 1.0)
    return output


if __name__ == "__main__":
    a = np.arange(50, step=2).reshape((1, 1, 5, 5))
    # print(scipy.ndimage.gaussian_filter(a, sigma=1))
    a = torch.tensor(a)
    f = torchvision.transforms.functional.gaussian_blur(a, (5, 5), sigma=5)
    print(f)
    print(gaussian_blur_torch(a, kernel_size=[5, 5], sigma=torch.tensor(5)))

    from skimage import data
    from skimage.filters import unsharp_mask as unsharp_mask_skimage
    import matplotlib.pyplot as plt
    from PIL import Image
    image = data.astronaut()  # moon()
    image = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(image)[None, :, :, :]
    img = torch.permute(img, (0, 3, 1, 2))
    img_one = torch.tensor(img.clone().detach(), requires_grad=True)
    img = torch.cat([img_one, img_one], dim=0)
    sigma = torch.tensor([[1.0]], requires_grad=True)
    amount = torch.tensor([[0.5], [10.0]], requires_grad=True)[:, :, None, None]

    # # test grad
    # res = unsharp_mask(img, sigma, amount)
    # res.mean().backward()
    # print(sigma.grad, amount.grad, img.grad)
    # result_3 = res.detach().cpu().numpy().squeeze(0)

    res = sharpness(img, amount)
    res.mean().backward()
    print(sigma.grad, amount.grad, img_one.grad)
    result_3 = res.detach().cpu().numpy()[0].transpose(1, 2, 0)

    res = adjust_sharpness(img, amount)
    result_5 = res.detach().cpu().numpy()[0].transpose(1, 2, 0)

    sharpness_adjuster = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1.0)
    result_4 = sharpness_adjuster(img).detach().cpu().numpy()[0].transpose(1, 2, 0)

    result_1 = unsharp_mask_skimage(image, radius=1, amount=0.5)
    result_2 = unsharp_mask_skimage(image, radius=1, amount=5)
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 10))
    ax = axes.ravel()
    ax[0].imshow(image)  # , cmap=plt.cm.gray
    ax[0].set_title('Original image')
    ax[1].imshow(result_1)
    ax[1].set_title('Enhanced image, radius=1, amount=1.0')
    ax[2].imshow(result_2)
    ax[2].set_title('Enhanced image, radius=5, amount=2.0')
    ax[3].imshow(result_3)
    ax[3].set_title('our')
    ax[4].imshow(result_4)
    ax[4].set_title('torchvision')
    # ax[5].imshow(image)
    # ax[5].set_title('original')
    ax[5].set_title('our2')
    ax[5].imshow(result_5)

    for a in ax:
        a.axis('off')
    fig.tight_layout()
    plt.show()
