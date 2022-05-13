import numpy as np
import torch
from torchvision.datasets import ImageFolder

from . import transforms as tf
from .base_dataset import BaseImageDataset


class ImagenetDataset(BaseImageDataset, ImageFolder):
    """
        Dataset class for the ImageNet dataset.
        Dataset structure
        + imagenet
          |- train
             |- n*
          |- val
             |- n*
        Both validation and training will have 1000 folders starting with 'n' (1 folder per class).
    """
    def __init__(self, opts, is_training = True, is_evaluation = False):
        """
            :param opts: arguments
            :param is_training: Training or validation mode
            :param is_evaluation: Evaluation mode
        """
        root = getattr(opts, "dataset.root_train", None) if is_training else getattr(opts, "dataset.root_val", None)
        BaseImageDataset.__init__(self, opts=opts, is_training=is_training, is_evaluation=is_evaluation)
        # root = self.root
        ImageFolder.__init__(self, root=root, transform=None, target_transform=None, is_valid_file=None)

    def training_transforms(self, size):
        """
            :param size: crop size (H, W)
            :return: list of augmentation methods
        """
        aug_list = [tf.RandomResizedCrop(opts=self.opts, size=size)]
        aug_list.extend(self.additional_transforms(opts=self.opts))
        aug_list.append(tf.NumpyToTensor(opts=self.opts))
        return tf.Compose(opts=self.opts, img_transforms=aug_list)

    def validation_transforms(self, size):
        """
            :param size: crop size (H, W)
            :return: list of augmentation methods
        """
        if isinstance(size, (tuple, list)):
            size = min(size)

        assert isinstance(size, int)
        # (256 - 224) = 32
        # where 224/0.875 = 256
        scale_size = size + 32 # int(make_divisible(crop_size / 0.875, divisor=32))

        return tf.Compose(opts=self.opts, img_transforms=[
            tf.Resize(opts=self.opts, size=scale_size),
            tf.CenterCrop(opts=self.opts, size=size),
            tf.NumpyToTensor(opts=self.opts)
        ])

    def evaluation_transforms(self, size):
        """
            :param size: crop size (H, W)
            :return: list of augmentation methods
        """
        return self.validation_transforms(size=size)

    def __getitem__(self, batch_indexes_tup):
        """
            :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
            :return: dictionary containing input image and label ID.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self.training_transforms(size=(crop_size_h, crop_size_w))
        else: # same for validation and evaluation
            transform_fn = self.validation_transforms(size=(crop_size_h, crop_size_w))

        img_path, target = self.samples[img_index]
        input_img = self.read_image(img_path)

        if input_img is None:
            # Sometimes images are corrupt and cv2 is not able to load them
            # Skip such images
            print('Img index {} is possibly corrupt. Removing it from the sample list'.format(img_index))
            del self.samples[img_index]
            input_img = np.zeros(shape=(crop_size_h, crop_size_w, 3), dtype=np.uint8)

        data = {"image": input_img}
        data = transform_fn(data)

        # target is a 0-dimensional tensor
        target_tensor = torch.tensor(1, dtype=torch.long).fill_(target)

        data["label"] = target_tensor
        return data

    def __len__(self):
        return len(self.samples)
