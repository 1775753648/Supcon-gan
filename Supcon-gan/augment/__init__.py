import gin
import torch
import torch.nn as nn
from kornia.filters import get_gaussian_kernel2d
from kornia.filters import filter2d as filter2D
from utils import call_with_accepted_args
from augment.color_jitter import *
from augment.spatial import *
from  augment.autoaugment import CIFAR10Policy
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

from third_party.diffaug import DiffAugment
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

@gin.configurable("augment", whitelist=["fn"])
def get_augment(mode='none', **kwargs):
    _mapping = {
        'none': NoAugment,
        'gaussian': Gaussian,
        'hflip': HorizontalFlipLayer,
        'hfrt': HorizontalFlipRandomCrop,
        'color_jitter': ColorJitterLayer,
        'cutout': CutOut,
        'simclr': simclr,
        'simclr_hq': simclr_hq,
        'simclr_hq_cutout': simclr_hq_cutout,
        'diffaug': diffaug,
        'simclr_normal': simclr_normal,
    }
    fn = _mapping[mode]
    return call_with_accepted_args(fn, **kwargs)


@gin.configurable
class NoAugment(nn.Module):
    def __init__(self):
        super(NoAugment, self).__init__()

    def forward(self, input):
        return input


@gin.configurable(whitelist=["sigma"])
class Gaussian(nn.Module):
    def __init__(self, sigma):
        super(Gaussian, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        output = input + torch.randn_like(input) * self.sigma
        output = output.clamp(0, 1)
        return output


@gin.configurable
class GaussianBlur(nn.Module):
    def __init__(self, sigma_range):
        """Blurs the given image with separable convolution.

        Args:
            sigma_range: Range of sigma for being used in each gaussian kernel.

        """
        super(GaussianBlur, self).__init__()
        self.sigma_range = sigma_range

    def forward(self, inputs):
        _device = inputs.device

        batch_size, num_channels, height, width = inputs.size()

        kernel_size = height // 10
        radius = int(kernel_size / 2)
        kernel_size = radius * 2 + 1

        sigma = np.random.uniform(*self.sigma_range)
        kernel = torch.unsqueeze(get_gaussian_kernel2d((kernel_size, kernel_size),
                                                       (sigma, sigma)), dim=0)
        blurred = filter2D(inputs, kernel, "reflect")

        return blurred


@gin.configurable
class RandomColorGrayLayer(nn.Module):
    def __init__(self):
        super(RandomColorGrayLayer, self).__init__()
        _weight = torch.tensor([[0.299, 0.587, 0.114]])
        self.register_buffer('_weight', _weight.view(1, 3, 1, 1))

    def forward(self, inputs):
        l = F.conv2d(inputs, self._weight)
        gray = torch.cat([l, l, l], dim=1)
        return gray


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, inputs):
        _prob = inputs.new_full((inputs.size(0),), self.p)
        _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
        return inputs * (1 - _mask) + self.fn(inputs) * _mask

class CustomTransforms(nn.Module):
    def __init__(self):
        super(CustomTransforms, self).__init__()

        self.train_trsf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize]
    )

    def forward(self, x):
        transformed_images = []
        for image in x:
            image_pil = to_pil_image(image)  # Convert each image to PIL format
            transformed_image = self.train_trsf(image_pil)
            transformed_images.append(transformed_image)

        # Convert the list of transformed images back to a tensor
        x_transformed = torch.stack(transformed_images, dim=0)
        x_transformed=x_transformed.to('cuda')
        return x_transformed
def simclr():

    return  CustomTransforms()



def simclr_normal():
    return nn.Sequential(
            RandomResizeCropLayer(),
            HorizontalFlipLayer(),
            RandomApply(ColorJitterLayer(), p=0.8),
            RandomApply(RandomColorGrayLayer(), p=0.2)
        )


def simclr_hq():
    return nn.Sequential(
            RandomResizeCropLayer(),
            HorizontalFlipLayer(),
            RandomApply(ColorJitterLayer(), p=0.8),
            RandomApply(RandomColorGrayLayer(), p=0.2),
            RandomApply(GaussianBlur(), p=0.5)
        )


def simclr_hq_cutout():
    return nn.Sequential(
            RandomResizeCropLayer(),
            HorizontalFlipLayer(),
            RandomApply(ColorJitterLayer(), p=0.8),
            RandomApply(RandomColorGrayLayer(), p=0.2),
            RandomApply(GaussianBlur(), p=0.5),
            RandomApply(CutOut(), p=0.5),
        )


class DiffAugLayer(nn.Module):
    def __init__(self, policy=""):
        super().__init__()
        self.policy = policy

    def forward(self, inputs):
        return DiffAugment(inputs, policy=self.policy)

def diffaug():
    return DiffAugLayer(policy='color,cutout')