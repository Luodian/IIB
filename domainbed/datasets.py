# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets.folder
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder, CIFAR10
from torchvision.transforms.functional import rotate
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "VerticalLine",
    "VHLine",
    "FullColoredMNIST",
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 4  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

class MultipleEnvironmentCIFAR10(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = CIFAR10(root, train=True, download=True)
        original_dataset_te = CIFAR10(root, train=False, download=True)

        original_images = np.concatenate((original_dataset_tr.data, original_dataset_te.data))
        original_labels = np.concatenate((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            self.datasets.append(dataset_transform(original_images, original_labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        self.colors = torch.FloatTensor(
            [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0], [0, 255, 0], [65, 105, 225], [0, 225, 225],
             [0, 0, 255], [255, 20, 147], [160, 160, 160]])
        self.random_colors = torch.randint(255, (10, 3)).float()

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class VHLine(MultipleEnvironmentCIFAR10):
    ENVIRONMENT_NAMES = [0, 1]
    N_WORKERS = 0
    N_STEPS = 10001

    def __init__(self, root, test_envs, hparams):
        self.domain_label = [0, 1]
        # print("MY COMBINE:", MY_COMBINE)
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        super(VHLine, self).__init__(root, self.domain_label, self.color_dataset, (3, 32, 32,), 10)

    def color_dataset(self, images, labels, environment):
        # Add a line to the last channel and vary its brightness during testing.
        images = self.add_vhline(images, labels, b_scale=1, env=environment)
        for i in range(5):
            rand_indx = np.random.randint(0, images.shape[0])
            self._plot(images[rand_indx])

        x = torch.Tensor(images).permute(0, 3, 1, 2)
        y = torch.Tensor(labels).view(-1).long()

        return TensorDataset(x, y)

    def add_vhline(self, images, labels, b_scale, env):
        images = np.divide(images, 255.0)
        if env == 1:
            return images

        def configurations(images, cond_indx, cls):
            # To create the ten-valued spurious feature, we consider a vertical line passing through the middle of each channel,
            # and also additionally the horizontal line through the first channel.
            if cls == 0:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 1:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 2:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 3:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 4:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 5:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 6:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 7:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 8:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
            elif cls == 9:
                images[cond_indx, :, 16:17, 0] = np.add(images[cond_indx, :, 16:17, 0], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 1] = np.add(images[cond_indx, :, 16:17, 1], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, :, 16:17, 2] = np.add(images[cond_indx, :, 16:17, 2], 0.5 + 0.5 * np.random.uniform(-b_scale, b_scale))
                images[cond_indx, 16:17, :, 0] = np.add(images[cond_indx, 16:17, :, 2], 0.5 - 0.5 * np.random.uniform(-b_scale, b_scale))

            return images

        for indx in range(self.num_classes):
            class_cond_index = (labels == indx)
            p_ii_arr = np.random.choice([True, False], p=[0.5, 0.5], size=class_cond_index.shape[0])
            class_cond_index = np.multiply(class_cond_index, p_ii_arr)
            images = configurations(images, class_cond_index, indx)
            for indx_j in range(self.num_classes):
                if indx_j != indx:
                    other_class_cond_index = (labels == indx_j)
                    p_ij_arr = np.random.choice([True, False], p=[0.05, 0.95], size=other_class_cond_index.shape[0])
                    other_class_cond_index = np.multiply(other_class_cond_index, p_ij_arr)
                    images = configurations(images, other_class_cond_index, indx_j)

        return images

    def _plot(self, img):
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class VerticalLine(MultipleEnvironmentCIFAR10):
    ENVIRONMENT_NAMES = [0, 1, 2, 3, 4, 5]

    def __init__(self, root, test_envs, hparams):
        self.scale = [0, -4, -2, 0, 2, 4]

        # print("MY COMBINE:", MY_COMBINE)
        super(VerticalLine, self).__init__(root, self.scale, self.color_dataset, (3, 32, 32,), 10)
        self.input_shape = (3, 32, 32)
        self.num_classes = 10

    def color_dataset(self, images, labels, environment):
        # Add a line to the last channel and vary its brightness during testing.
        images = self.add_line(images, environment)
        # for i in range(5):
        #     rand_indx = np.random.randint(0, images.shape[0])
        #     self._plot(images[rand_indx])
        # images = torch.stack([images, images], dim=1)

        x = torch.Tensor(images).permute(0, 3, 1, 2)
        y = torch.Tensor(labels).view(-1).long()

        return TensorDataset(x, y)

    def add_line(self, images, b):
        images = np.divide(images, 255.0)
        # add 4 to last channel to avoid negative values in this channel.
        images[:, :, :, 2] = np.add(images[:, :, :, 2], 4)
        images[:, :, 16:17, 2] = np.add(images[:, :, 16:17, 2], np.float(b))
        images[:, :, :, 2] = np.divide(images[:, :, :, 2], 9)
        return images

    def _plot(self, img):
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class FullColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENT_NAMES = [0, 1, 2]

    def __init__(self, root, test_envs, hparams):
        self.data_type = hparams['type']
        if self.data_type == 0:
            self.ratio = hparams.get('ratio', 0.9)
            self.env_seed = hparams.get('env_seed', 1)
            MY_COMBINE = [[self.env_seed, True, 0.0], [self.env_seed, True, 1.0], [self.env_seed, True, self.ratio]]
        else:
            raise NotImplementedError

        # print("MY COMBINE:", MY_COMBINE)
        super(FullColoredMNIST, self).__init__(root, MY_COMBINE, self.color_dataset, (3, 28, 28,), 10)
        self.input_shape = (3, 28, 28)
        self.num_classes = 10

    def color_dataset(self, images, labels, environment):
        # set the seed
        original_seed = torch.cuda.initial_seed()
        torch.manual_seed(environment[0])
        shuffle = torch.randperm(len(self.colors))
        self.colors_ = self.colors[shuffle] if environment[1] else self.random_colors[shuffle]
        torch.manual_seed(environment[0])
        ber = self.torch_bernoulli_(environment[2], len(labels))
        print("ber:", len(ber), sum(ber))
        torch.manual_seed(original_seed)

        images = torch.stack([images, images, images], dim=1)
        # binarize the images
        images = (images > 0).float()
        y = labels.view(-1).long()
        color_label = torch.zeros_like(y).long()

        # Apply the color to the image
        for img_idx in range(len(images)):
            if ber[img_idx] > 0:
                color_label[img_idx] = labels[img_idx]
                for channels in range(3):
                    images[img_idx, channels, :, :] = images[img_idx, channels, :, :] * \
                                                      self.colors_[labels[img_idx].long(), channels]
            else:
                color = torch.randint(10, [1])[0]  # random color, regardless of label
                color_label[img_idx] = color
                for channels in range(3):
                    images[img_idx, channels, :, :] = images[img_idx, channels, :, :] * self.colors_[color, channels]

        x = images.float().div_(255.0)
        for i in range(5):
            rand_indx = np.random.randint(0, x.shape[0])
            self._plot(images[rand_indx])
        return TensorDataset(True, x, y, color_label)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

    def _plot(self, img):
        plt.imshow(torch.permute(img, (1, 2, 0)))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    N_WORKERS = 0
    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class Spirals(MultipleDomainDataset):
    CHECKPOINT_FREQ = 10
    ENVIRONMENTS = [str(i) for i in range(16)]

    def __init__(self, root, test_env, hparams):
        super().__init__()
        self.datasets = []

        test_dataset = self.make_tensor_dataset(env='test')
        self.datasets.append(test_dataset)
        for env in self.ENVIRONMENTS:
            env_dataset = self.make_tensor_dataset(env=env, seed=int(env))
            self.datasets.append(env_dataset)

        self.input_shape = (18,)
        self.num_classes = 2

    def make_tensor_dataset(self, env, n_examples=1024, n_envs=16, n_revolutions=3, n_dims=16,
                            flip_first_signature=False,
                            seed=0):

        if env == 'test':
            inputs, labels = self.generate_environment(2000,
                                                       n_rotations=n_revolutions,
                                                       env=env,
                                                       n_envs=n_envs,
                                                       n_dims_signatures=n_dims,
                                                       seed=2 ** 32 - 1
                                                       )
        else:
            inputs, labels = self.generate_environment(n_examples,
                                                       n_rotations=n_revolutions,
                                                       env=env,
                                                       n_envs=n_envs,
                                                       n_dims_signatures=n_dims,
                                                       seed=seed
                                                       )
        if flip_first_signature:
            inputs[:1, 2:] = -inputs[:1, 2:]

        return TensorDataset(torch.tensor(inputs), torch.tensor(labels))

    def generate_environment(self, n_examples, n_rotations, env, n_envs,
                             n_dims_signatures,
                             seed=None):
        """
        env must either be "test" or an int between 0 and n_envs-1
        n_dims_signatures: how many dimensions for the signatures (spirals are always 2)
        seed: seed for numpy
        """
        assert env == 'test' or 0 <= int(env) < n_envs

        # Generate fixed dictionary of signatures
        rng = np.random.RandomState(seed)

        signatures_matrix = rng.randn(n_envs, n_dims_signatures)

        radii = rng.uniform(0.08, 1, n_examples)
        angles = 2 * n_rotations * np.pi * radii

        labels = rng.randint(0, 2, n_examples)
        angles = angles + np.pi * labels

        radii += rng.uniform(-0.02, 0.02, n_examples)
        xs = np.cos(angles) * radii
        ys = np.sin(angles) * radii

        if env == 'test':
            signatures = rng.randn(n_examples, n_dims_signatures)
        else:
            env = int(env)
            signatures_labels = np.array(labels * 2 - 1).reshape(1, -1)
            signatures = signatures_matrix[env] * signatures_labels.T

        signatures = np.stack(signatures)
        mechanisms = np.stack((xs, ys), axis=1)
        mechanisms /= mechanisms.std(axis=0)  # make approx unit variance (signatures already are)
        inputs = np.hstack((mechanisms, signatures))

        return inputs.astype(np.float32), labels.astype(np.long)
