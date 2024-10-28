import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(torch.nn.Module):
    def __init__(self, shape=(17, 64, 64), mid_channels=32, output_dim=4096):
        """shape: c,h,w"""
        super(FeatureExtractor, self).__init__()
        in_channels = shape[0]
        self.output_dim = output_dim

        min_feature_map_size = 4
        assert output_dim % (min_feature_map_size ** 2) == 0, 'output dim=%d' % output_dim
        size = int(shape[2])
        # print('Agent CNN:')
        # print('    ', shape)
        size = size // 2
        channels = mid_channels
        layers = []
        layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        while size > min_feature_map_size:
            in_channels = channels
            if size == min_feature_map_size * 2:
                channels = output_dim // (min_feature_map_size ** 2)
            else:
                channels *= 2
            assert size % 2 == 0
            size = size // 2
            # print(size, in_channels, channels)
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            # layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            # layers.append(nn.BatchNorm2d(channels))
            # layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = torch.reshape(x, [-1, self.output_dim])
        return x


# input: float in [0, 1]
class Value(nn.Module):
    def __init__(self, cfg, shape=(19, 64, 64)):
        super(Value, self).__init__()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(shape=shape, mid_channels=cfg.base_channels,
                                                  output_dim=cfg.feature_extractor_dims)

        self.fc1 = nn.Linear(cfg.feature_extractor_dims, cfg.fc1_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(cfg.fc1_size, 1)
        self.tanh = nn.Tanh()

        self.down_sample = nn.AdaptiveAvgPool2d((shape[1], shape[2]))

    def forward(self, images, states=None):
        images = self.down_sample(images)
        lum = (images[:, 0, :, :] * 0.27 + images[:, 1, :, :] * 0.67 + images[:, 2, :, :] * 0.06 + 1e-5)[:, None, :, :]
        # print(lum.shape)
        # luminance and contrast
        luminance = torch.mean(lum, dim=(1, 2, 3))
        contrast = torch.var(lum, dim=(1, 2, 3))
        # saturation
        i_max, _ = torch.max(torch.clip(images, min=0.0, max=1.0), dim=1)
        i_min, _ = torch.min(torch.clip(images, min=0.0, max=1.0), dim=1)
        # print("i_max i_min shape:", i_max.shape, i_min.shape)
        sat = (i_max - i_min) / (torch.minimum(i_max + i_min, 2.0 - i_max - i_min) + 1e-2)
        # print("sat.shape", sat.shape)
        saturation = torch.mean(sat, dim=[1, 2])
        # print("luminance shape:", luminance.shape, contrast.shape, saturation.shape)
        repetition = 1
        state_feature = torch.cat(
            [torch.tile(luminance[:, None], [1, repetition]),
             torch.tile(contrast[:, None], [1, repetition]),
             torch.tile(saturation[:, None], [1, repetition])], dim=1)
        # print('States:', states.shape)
        if states is None:
            states = state_feature
        else:
            assert len(states.shape) == len(state_feature.shape)
            states = torch.cat([states, state_feature], dim=1)
        if states is not None:
            states = states[:, :, None, None] + images[:, 0:1, :, :] * 0
            # print('     States:', states.shape)
            images = torch.cat([images, states], dim=1)
            # print("images.shape", images.shape)
        feature = self.feature_extractor(images)
        # print('     CNN shape: ', feature.shape)
        # print('Before final FCs', feature.shape)
        out = self.fc2(self.lrelu(self.fc1(feature)))
        # print('     ', out.shape)
        # out = self.tanh(out)
        return out


if __name__ == "__main__":
    from easydict import EasyDict
    import numpy as np
    cfg = EasyDict()
    cfg['base_channels'] = 32
    cfg['fc1_size'] = 128
    cfg['feature_extractor_dims'] = 4096

    np.random.seed(0)
    x = torch.randn((1, 3, 512, 512))
    states = torch.randn((1, 11))
    # x = np.transpose(x, (0, 3, 1, 2))
    # x = torch.from_numpy(x)
    value = Value(cfg)
    y = value(x, states)
    print(y.shape, y)
    print(value.state_dict())
    torch.save(value.state_dict(), "value.pth")