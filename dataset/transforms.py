import cv2
import math
import numpy as np
import random
import torch
from typing import Sequence

"""
    Different image transformation functions are defined in this file. We use OpenCV and not PIL Image
"""

_str_to_cv2_interpolation = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
}

_cv2_to_str_interpolation = {
    cv2.INTER_NEAREST: 'nearest',
    cv2.INTER_LINEAR: 'bilinear',
    cv2.INTER_CUBIC: 'cubic'
}

_str_to_cv2_pad = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}



class BaseTransformation(object):
    """
        Base class for transformations
    """
    def __init__(self, opts):
        super(BaseTransformation, self).__init__()
        self.opts = opts

    def __call__(self, data):
        raise NotImplementedError

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)



def _cv2_interpolation(interpolation):
    if interpolation not in _str_to_cv2_interpolation:
        interpolate_modes = list(_str_to_cv2_interpolation.keys())
        inter_str = "Supported interpolation modes are:"
        for i, j in enumerate(interpolate_modes):
            inter_str += "\n\t{}: {}".format(i, j)
        raise ValueError(inter_str)
    return _str_to_cv2_interpolation[interpolation]


def _cv2_padding(pad_mode):
    if pad_mode not in _str_to_cv2_pad:
        pad_modes = list(_str_to_cv2_pad.keys())
        pad_mode_str = "Supported padding modes are:"
        for i, j in enumerate(pad_modes):
            pad_mode_str += "\n\t{}: {}".format(i, j)
        raise ValueError(pad_mode_str)
    return _str_to_cv2_pad[pad_mode]


def _crop_fn(data, i, j, h, w):
    img = data["image"]
    crop_image = img[i:i + h, j:j + w]
    data["image"] = crop_image

    mask = data.get("mask", None)
    if mask is not None:
        crop_mask = mask[i:i + h, j:j + w]
        data["mask"] = crop_mask
    return data


def _resize_fn(data, size, interpolation="bilinear"):
    img = data["image"]

    if isinstance(size, Sequence) and len(size) == 2:
        size_h, size_w = size[0], size[1]
    elif isinstance(size, int):
        h, w, _ = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return data

        if w < h:
            size_h = int(size * h / w)

            size_w = size
        else:
            size_w = int(size * w / h)
            size_h = size
    else:
        raise TypeError(
            'Supported size args are int or tuple of length 2. Got inappropriate size arg: {}'.format(size)
        )
    if isinstance(interpolation, str):
        interpolation = _str_to_cv2_interpolation[interpolation]
    img = cv2.resize(img, dsize=(size_w, size_h), interpolation=interpolation)
    data["image"] = img

    mask = data.get("mask", None)
    if mask is not None:
        mask = cv2.resize(mask, dsize=(size_w, size_h), interpolation=cv2.INTER_NEAREST)
        data["mask"] = mask

    return data


def setup_size(size, error_msg="Need a tuple of length 2"):
    if isinstance(size, int):
        return size, size

    if isinstance(size, (list, tuple)) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomGammaCorrection(BaseTransformation):
    def __init__(self, opts):
        gamma_range = getattr(opts, "image_augmentation.random_gamma_correction.gamma", (0.25, 1.75))
        p = getattr(opts, "image_augmentation.random_gamma_correction.p", 0.5)
        super(RandomGammaCorrection, self).__init__(opts=opts)
        self.gamma = setup_size(gamma_range)
        self.p = p

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.random-gamma-correction.enable", action="store_true",
                           help="use gamma correction")
        group.add_argument("--image-augmentation.random-gamma-correction.gamma", type=float or tuple,
                           default=(0.5, 1.5), help="Gamma range")
        group.add_argument("--image-augmentation.random-gamma-correction.p", type=float,
                           default=0.5, help="Probability that {} will be applied".format(cls.__name__))
        return parser

    def __call__(self, data):
        if random.random() <= self.p:
            img = data["image"]
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
            data["image"] = img
        return data

    def __repr__(self):
        return '{}(gamma={}, p={})'.format(self.__class__.__name__, self.gamma, self.p)


class RandomResize(BaseTransformation):
    def __init__(self, opts):
        min_size = getattr(opts, "image_augmentation.random_resize.min_size", 256)
        max_size = getattr(opts, "image_augmentation.random_resize.max_size", 1024)
        interpolation = getattr(opts, "image-augmentation.random_resize.interpolation", "bilinear")
        super(RandomResize, self).__init__(opts=opts)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = _cv2_interpolation(interpolation=interpolation)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-resize.enable", action="store_true", help="Use random resize")
        group.add_argument("--image-augmentation.random-resize.min-size", type=int, default=256, help="Min size")
        group.add_argument("--image-augmentation.random-resize.max-size", type=int, default=1024, help="Max size")
        group.add_argument("--image-augmentation.random-resize.interpolation", type=str, default="bilinear",
                           help="Interpolation method")
        return parser

    def __call__(self, data):
        random_size = random.randint(self.min_size, self.max_size)
        return _resize_fn(data, size=random_size, interpolation=self.interpolation)

    def __repr__(self):
        return '{}(min_size={}, max_size={}, interpolation={})'.format(
            self.__class__.__name__,
            self.min_size,
            self.max_size,
            _cv2_to_str_interpolation[self.interpolation]
        )


class RandomZoomOut(BaseTransformation):
    def __init__(self, opts, size=None):
        side_range = getattr(opts, "image_augmentation.random_zoom_out.side_range", [1, 4])
        p = getattr(opts, "image_augmentation.random_zoom_out.p", 0.5)
        super(RandomZoomOut, self).__init__(opts=opts)
        self.fill = 0.5
        self.side_range = side_range
        self.p = p

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-zoom-out.enable", action="store_true", help="Use random scale")
        group.add_argument("--image-augmentation.random-zoom-out.side-range", type=list or tuple, default=[1, 4], 
                           help="Side range")
        group.add_argument("--image-augmentation.random-zoom-out.p", type=float, default=0.5,
                           help="Probability of applying RandomZoomOut transformation")
        return parser

    def zoom_out(self, image, boxes=None):
        height, width, depth = image.shape
        ratio = random.uniform(self.side_range[0], self.side_range[1])
        left = int(random.uniform(0, width * ratio - width))
        top = int(random.uniform(0, height * ratio - height))

        expand_image = np.ones((int(height * ratio), int(width * ratio), depth), dtype=image.dtype) * self.fill
        expand_image[top:top + height, left:left + width] = image

        expand_boxes = None
        if boxes is not None:
            expand_boxes = boxes.copy()
            expand_boxes[:, :2] += (left, top)
            expand_boxes[:, 2:] += (left, top)

        return expand_image, expand_boxes

    def __call__(self, data):
        if random.random() > self.p:
            return data
        img = data["image"]
        boxes = data.get("box_coordinates", None)

        img, boxes = self.zoom_out(image=img, boxes=boxes)

        data["image"] = img
        data["box_coordinates"] = boxes

        return data

    def __repr__(self):
        return '{}(min_scale={}, max_scale={}, interpolation={})'.format(
            self.__class__.__name__,
            self.min_scale,
            self.max_scale,
            _cv2_to_str_interpolation[self.interpolation]
        )


class RandomScale(BaseTransformation):
    def __init__(self, opts, size=None):
        min_scale = getattr(opts, "image_augmentation.random_scale.min_scale", 0.5)
        max_scale = getattr(opts, "image_augmentation.random_scale.max_scale", 2.0)
        interpolation = getattr(opts, "image_augmentation.random_scale.interpolation", "bilinear")
        super(RandomScale, self).__init__(opts=opts)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = _cv2_interpolation(interpolation)
        self.size = None
        if size is not None:
            self.size = setup_size(size)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-scale.enable", action="store_true", help="Use random scale")
        group.add_argument("--image-augmentation.random-scale.min-scale", type=float, default=0.5, help="Min scale")
        group.add_argument("--image-augmentation.random-scale.max-scale", type=float, default=2.0, help="Max scale")
        group.add_argument("--image-augmentation.random-scale.interpolation", type=str, default="bilinear",
                           help="Interpolation method")
        return parser

    def __call__(self, data):
        scale = random.uniform(self.min_scale, self.max_scale)

        img = data["image"]
        if self.size is None:
            height, width = img.shape[:2]
        else:
            height, width = self.size
        target_height, target_width = int(height * scale), int(width * scale)
        img = cv2.resize(img, dsize=(target_width, target_height), interpolation=self.interpolation)
        data["image"] = img

        mask = data.get("mask", None)
        if mask is not None:
            mask = cv2.resize(mask, dsize=(target_width, target_height), interpolation=cv2.INTER_NEAREST)
            data["mask"] = mask
        return data

    def __repr__(self):
        return '{}(min_scale={}, max_scale={}, interpolation={})'.format(
            self.__class__.__name__,
            self.min_scale,
            self.max_scale,
            _cv2_to_str_interpolation[self.interpolation]
        )


class RandomResizedCrop(BaseTransformation):
    """
        Adapted from Pytorch Torchvision
    """
    def __init__(self, opts, size):

        interpolation = getattr(opts, "image_augmentation.random_resized_crop.interpolation", "bilinear")
        scale = getattr(opts, "image_augmentation.random_resized_crop.scale", (0.08, 1.0))
        ratio = getattr(opts, "image_augmentation.random_resized_crop.aspect_ratio", (3. / 4., 4. / 3.))

        if not isinstance(scale, Sequence) or (
                isinstance(scale, Sequence) and len(scale) != 2 and 0.0 <= scale[0] < scale[1]
        ):
            raise ValueError(
                "--image-augmentation.random-resized-crop.scale should be a tuple of length 2 "
                "such that 0.0 <= scale[0] < scale[1]. Got: {}".format(scale)
            )

        if not isinstance(ratio, Sequence) or (
                isinstance(ratio, Sequence) and len(ratio) != 2 and 0.0 < ratio[0] < ratio[1]
        ):
            raise ValueError(
                "--image-augmentation.random-resized-crop.aspect-ratio should be a tuple of length 2 "
                "such that 0.0 < ratio[0] < ratio[1]. Got: {}".format(ratio)
            )

        ratio = (round(ratio[0], 3), round(ratio[1], 3))

        super(RandomResizedCrop, self).__init__(opts=opts)

        self.scale = scale
        self.size = setup_size(size=size)

        self.interpolation = _cv2_interpolation(interpolation)
        self.ratio = ratio

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.random-resized-crop.enable", action="store_true",
                           help="use gamma correction")

        group.add_argument("--image-augmentation.random-resized-crop.interpolation", type=str,
                           default="bilinear", choices=list(_str_to_cv2_interpolation.keys()),
                           help="Interpolation for resizing")
        group.add_argument("--image-augmentation.random-resized-crop.scale", type=tuple,
                           default=(0.08, 1.0),
                           help="scale range of the cropped image before resizing, relatively to the origin image")
        group.add_argument("--image-augmentation.random-resized-crop.aspect-ratio", type=float or tuple,
                           default=(3. / 4., 4. / 3.),
                           help="Aspect ratio range of the cropped image before resizing.")
        return parser

    def get_params(self, height, width):
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = (1.0 * width) / height
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data):
        img = data["image"]
        height, width = img.shape[:2]

        i, j, h, w = self.get_params(height=height, width=width)
        data = _crop_fn(data=data, i=i, j=j, h=h, w=w)
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self):
        return '{}(scale={}, ratio={}, interpolation={})'.format(
            self.__class__.__name__,
            self.scale,
            self.ratio,
            _cv2_to_str_interpolation[self.interpolation]
        )


class RandomCrop(BaseTransformation):
    """
        Randomly crop the image to a given size
    """
    def __init__(self, opts, size):
        super(RandomCrop, self).__init__(opts=opts)
        self.height, self.width = setup_size(size=size)
        self.opts = opts
        self.fill_mask = getattr(opts, "image_augmentation.random_crop.mask_fill", 255)
        is_padding = not getattr(opts, "image_augmentation.random_crop.resize_if_needed", False)
        self.inp_process_fn = self.pad_if_needed if is_padding else self.resize_if_needed


    @staticmethod
    def get_params(img_h, img_w, target_h, target_w):
        if img_w == target_w and img_h == target_h:
            return 0, 0, img_h, img_w
        i = random.randint(0, img_h - target_h)
        j = random.randint(0, img_w - target_w)
        return i, j, target_h, target_w

    def pad_if_needed(self, data):
        img = data["image"]

        h, w, channels = img.shape
        pad_h = self.height - h if h < self.height else 0
        pad_w = self.width - w if w < self.width else 0

        # padding format is (top, bottom, left, right)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        data["image"] = img

        mask = data.get("mask", None)
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.fill_mask)
            data["mask"] = mask
        return data

    def resize_if_needed(self, data):
        img = data["image"]

        h, w, channels = img.shape
        new_size = min(
            h + max(0, self.height - h),
            w + max(0, self.width - w)
        )
        # resize while maintaining the aspect ratio
        return _resize_fn(data, size=new_size, interpolation="bilinear")

    def __call__(self, data):

        data = self.inp_process_fn(data)

        img_h, img_w = data["image"].shape[:2]
        i, j, h, w = self.get_params(img_h=img_h, img_w=img_w, target_h=self.height, target_w=self.width)
        data = _crop_fn(data=data, i=i, j=j, h=h, w=w)

        return data

    def __repr__(self):
        return '{}(size=(h={}, w={}))'.format(self.__class__.__name__, self.height, self.width)


class RandomFlip(BaseTransformation):
    def __init__(self, opts):
        super(RandomFlip, self).__init__(opts=opts)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.random-flip.enable", action="store_true",
                           help="use random flipping")
        return parser

    def __call__(self, data):
        flip_choice = random.choices([0, 1, 2])[0]
        if flip_choice in [0, 1]:  # 1 - Horizontal, 0 - vertical
            img = data["image"]
            img = cv2.flip(img, flip_choice)
            data["image"] = img

            mask = data.get("mask", None)
            if mask is not None:
                mask = cv2.flip(mask, flip_choice)
                data["mask"] = mask

            boxes = data.get("box_coordinates", None)
            if boxes is not None:
                boxes = boxes.copy()
                if flip_choice == 0:
                    height = img.shape[0]
                    boxes[:, 1::2] = height - boxes[:, 3::-2]
                elif flip_choice == 1:
                    width = img.shape[1]
                    boxes[:, 0::2] = width - boxes[:, 2::-2]

                data["box_coordinates"] = boxes

        return data


class RandomHorizontalFlip(BaseTransformation):
    def __init__(self, opts):
        p = getattr(opts, "image_augmentation.random_horizontal_flip.p", 0.5)
        super(RandomHorizontalFlip, self).__init__(opts=opts)
        self.p = p

    def __call__(self, data):

        if random.random() <= self.p:
            img = data["image"]
            img = cv2.flip(img, 1)
            data["image"] = img

            mask = data.get("mask", None)
            if mask is not None:
                mask = cv2.flip(mask, 1)
                data["mask"] = mask

            boxes = data.get("box_coordinates", None)
            if boxes is not None:
                boxes = boxes.copy()
                width = img.shape[1]
                boxes[:, 0::2] = width - boxes[:, 2::-2]

                data["box_coordinates"] = boxes

        return data

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-horizontal-flip.enable", action="store_true",
                           help="use random horizontal flipping")
        group.add_argument("--image-augmentation.random-horizontal-flip.p", type=float,
                           default=0.5, help="Probability for random horizontal flip")
        return parser

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class RandomVerticalFlip(BaseTransformation):
    def __init__(self, opts):
        p = getattr(opts, "image_augmentation.random_vertical_flip.p", 0.5)
        super(RandomVerticalFlip, self).__init__(opts=opts)
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            img = data["image"]
            img = cv2.flip(img, 0)
            data["image"] = img

            mask = data.get("mask", None)
            if mask is not None:
                mask = cv2.flip(mask, 0)
                data["mask"] = mask

            boxes = data.get("box_coordinates", None)
            if boxes is not None:
                boxes = boxes.copy()
                height = img.shape[0]
                boxes[:, 1::2] = height - boxes[:, 3::-2]

                data["box_coordinates"] = boxes

        return data

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-vertical-flip.enable", action="store_true",
                           help="use random vertical flipping")
        group.add_argument("--image-augmentation.random-vertical-flip.p", type=float,
                           default=0.5, help="Probability for random vertical flip")
        return parser

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class RandomRotate(BaseTransformation):
    def __init__(self, opts):
        angle = getattr(opts, "image_augmentation.random_rotate.angle", 10.0)
        fill = getattr(opts, "image_augmentation.random_rotate.fill_mask", 255)
        interpolation = getattr(opts, "image_augmentation.random_rotate.interpolation", "bilinear")
        p = getattr(opts, "image_augmentation.random_rotate.p", 0.5)
        super(RandomRotate, self).__init__(opts=opts)
        self.angle = angle
        self.fill = fill
        self.p = p
        self.interpolation = _cv2_interpolation(interpolation)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-rotate.enable", action="store_true",
                           help="use random rotation")

        group.add_argument("--image-augmentation.random-rotate.angle", type=float,
                           default=10.0, help="Angle is uniformly sampled from (-a, a)")
        group.add_argument("--image-augmentation.random-rotate.fill-mask", type=int or tuple,
                           default=255, help="Value used to fill the area after rotation")
        group.add_argument("--image-augmentation.random-rotate.interpolation", type=str,
                           default="bilinear", help="Interpolation method")
        group.add_argument("--image-augmentation.random-rotate.p", type=float,
                           default=0.5, help="Probability that {} will be applied".format(cls.__name__))
        return parser

    def __call__(self, data):
        img = data["image"]
        height, width = img.shape[:2]

        random_angle = random.uniform(-self.angle, self.angle)
        rotation_mat = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=random_angle, scale=1)

        img_rotated = cv2.warpAffine(
            src=img, M=rotation_mat, dsize=(width, height), flags=self.interpolation, borderValue=0
        )
        data["image"] = img_rotated

        mask = data["mask"]
        if mask is not None:
            mask_rotated = cv2.warpAffine(
                src=mask, M=rotation_mat, dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=self.fill
            )
            data["mask"] = mask_rotated

        return data

    def __repr__(self):
        return '{}(angle={}, interpolation={}, p={})'.format(
            self.__class__.__name__,
            self.angle,
            _cv2_to_str_interpolation[self.interpolation],
            self.p
        )


BLUR_METHODS = ['gauss', 'median', 'average', 'none', 'any']


class RandomBlur(BaseTransformation):
    def __init__(self, opts):
        kernel_range = getattr(opts, "image_augmentation.random_blur.kernel_size", [3, 7])
        blur_type = getattr(opts, "image_augmentation.random_blur.kernel_type", "any")
        p = getattr(opts, "image_augmentation.random_blur.p", 0.5)
        super(RandomBlur, self).__init__(opts=opts)
        self.kernel_range = setup_size(kernel_range)
        assert 1 <= self.kernel_range[0] <= self.kernel_range[1], "Got: {}".format(self.kernel_range)
        self.blur_type = blur_type
        self.p = p

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.random-blur.enable", action="store_true",
                           help="use random blurring")

        group.add_argument("--image-augmentation.random-blur.kernel-size", type=tuple or int or list,
                           default=[3, 7], help="Randomly sample the kernel size from the given range")
        group.add_argument("--image-augmentation.random-blur.kernel-type", type=str, choices=BLUR_METHODS,
                           default=255, help="Value used to fill the area after rotation")
        group.add_argument("--image-augmentation.random-blur.p", type=float,
                           default=0.5, help="Probability that {} will be applied".format(cls.__name__))
        return parser

    def blur_median(self, img, ksize_x, ksize_y):
        ksize = ksize_x if random.random() < 0.5 else ksize_y
        img = cv2.medianBlur(src=img, ksize=ksize)
        return img

    def blur_avg(self, img, ksize_x, ksize_y):
        return cv2.blur(src=img, ksize=(ksize_x, ksize_y))

    def blur_gauss(self, img, ksize_x, ksize_y):
        return cv2.GaussianBlur(src=img, ksize=(ksize_x, ksize_y), sigmaX=0)

    def blur_any(self, img, ksize_x, ksize_y):
        blur_method = random.choice(BLUR_METHODS[:-1])
        if blur_method == 'gauss':
            img = self.blur_gauss(img=img, ksize_x=ksize_x, ksize_y=ksize_y)
        elif blur_method == 'median':
            img = self.blur_median(img=img, ksize_x=ksize_x, ksize_y=ksize_y)
        elif blur_method == 'average':
            img = self.blur_avg(img=img, ksize_x=ksize_x, ksize_y=ksize_y)
        return img

    def __call__(self, data):
        if self.blur_type == 'none':
            return data

        ksize_x = random.randint(self.kernel_range[0], self.kernel_range[1])
        ksize_y = random.randint(self.kernel_range[0], self.kernel_range[1])
        ksize_x = (ksize_x // 2) * 2 + 1
        ksize_y = (ksize_y // 2) * 2 + 1

        img = data["image"]

        if self.blur_type == 'any':
            img = self.blur_any(img, ksize_x=ksize_x, ksize_y=ksize_y)
        elif self.blur_type == 'gaussian':
            img = self.blur_gauss(img=img, ksize_x=ksize_x, ksize_y=ksize_y)
        elif self.blur_type == 'median':
            img = self.blur_median(img=img, ksize_x=ksize_x, ksize_y=ksize_y)
        elif self.blur_type == 'average':
            img = self.blur_avg(img=img, ksize_x=ksize_x, ksize_y=ksize_y)

        data["image"] = img
        return data

    def __repr__(self):
        if self.blur_type == 'any':
            blur_type = ['gaussian', 'median', 'average']
        else:
            blur_type = self.blur_type
        return '{}(blur_type={}, kernel_range={})'.format(self.__class__.__name__, blur_type, self.kernel_range)


class RandomTranslate(BaseTransformation):
    def __init__(self, opts):
        translate_factor = getattr(opts, "image_augmentation.random_translate.factor", 0.2)
        assert 0 < translate_factor < 0.5, 'Factor should be between 0 and 0.5'
        super(RandomTranslate, self).__init__(opts=opts)

        self.translation_factor = translate_factor

    def __call__(self, data):
        img = data["image"]

        height, width = img.shape[:2]
        th = int(math.ceil(random.uniform(0, self.translation_factor) * height))
        tw = int(math.ceil(random.uniform(0, self.translation_factor) * width))
        img_translated = np.zeros_like(img)
        translate_from_left = True if random.random() <= 0.5 else False
        if translate_from_left:
            img_translated[th:, tw:] = img[:height - th, : width - tw]
        else:
            img_translated[:height - th, : width - tw] = img[th:, tw:]
        data["image"] = img_translated

        mask = data.get("mask", None)
        if mask is not None:
            mask_translated = np.zeros_like(mask)
            if translate_from_left:
                mask_translated[th:, tw:] = mask[:height - th, : width - tw]
            else:
                mask_translated[:height - th, : width - tw] = mask[th:, tw:]
            data["mask"] = mask_translated
        return data

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-translate.enable", action="store_true",
                           help="use random translation")
        group.add_argument("--image-augmentation.random-translate.factor", type=float,
                           default=0.2, help="Translate uniformly between (-u, u)")
        return parser

    def __repr__(self):
        return '{}(factor={})'.format(self.__class__.__name__, self.translation_factor)


class Resize(BaseTransformation):
    def __init__(self, opts, size, *args, **kwargs):
        if not (
                isinstance(size, int)
                or
                (isinstance(size, Sequence) and len(size) in (1, 2))
        ):
            raise TypeError(
                'Supported size args are int or tuple of length 2. Got inappropriate size arg: {}'.format(self.size)
            )
        interpolation = getattr(opts, "image_augmentation.resize.interpolation", "bilinear")
        super(Resize, self).__init__(opts=opts)

        self.size = size
        self.interpolation = _cv2_interpolation(interpolation)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.resize.enable", action="store_true",
                           help="use fixed resizing")

        group.add_argument("--image-augmentation.resize.interpolation", type=str, default="bilinear",
                           choices=list(_str_to_cv2_interpolation.keys()),
                           help="Interpolation for resizing. Default is bilinear")
        group.add_argument("--image-augmentation.resize.no-maintain-aspect-ratio", action="store_true",
                           help="Maintain aspect ratio while resizing. Default is True.")
        return parser

    def __call__(self, data):
        return _resize_fn(data=data, size=self.size, interpolation=self.interpolation)

    def __repr__(self):
        return '{}(size={}, interpolation={})'.format(
            self.__class__.__name__,
            self.size,
            _cv2_to_str_interpolation[self.interpolation]
        )


class CenterCrop(BaseTransformation):
    def __init__(self, opts, size):
        super(CenterCrop, self).__init__(opts=opts)
        if isinstance(size, Sequence) and len(size) == 2:
            self.height, self.width = size[0], size[1]
        elif isinstance(size, Sequence) and len(size) == 1:
            self.height = self.width = size[0]
        elif isinstance(size, int):
            self.height = self.width = size
        else:
            raise ValueError('Scale should be either an int or tuple of ints')

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))

        group.add_argument("--image-augmentation.center-crop.enable", action="store_true",
                           help="use center cropping")
        return parser

    def __call__(self, data):
        height, width = data["image"].shape[:2]
        i = (height - self.height) // 2
        j = (width - self.width) // 2
        return _crop_fn(data=data, i=i, j=j, h=self.height, w=self.width)

    def __repr__(self):
        return '{}(size=(h={}, w={}))'.format(self.__class__.__name__, self.height, self.width)


class RandomJPEGCompress(BaseTransformation):
    def __init__(self, opts):
        q_range = getattr(opts, "image_augmentation.random_jpeg_compress.q_factor", (5, 25))
        if isinstance(q_range, (int, float)):
            q_range = (max(q_range - 10, 0), q_range)
        assert len(q_range) == 2
        assert q_range[0] <= q_range[1]
        p = getattr(opts, "image_augmentation.random_jpeg_compress.p", 0.5)
        super(RandomJPEGCompress, self).__init__(opts=opts)
        self.q_factor = q_range
        self.p = p

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-jpeg-compress.enable", action="store_true",
                           help="use random compression")
        group.add_argument("--image-augmentation.random-jpeg-compress.q-factor", type=int or tuple, default=(5, 25),
                           help="Compression quality factor range")
        group.add_argument("--image-augmentation.random-jpeg-compress.p", type=float,
                           default=0.5, help="Probability that {} will be applied".format(cls.__name__))

        return parser

    def __call__(self, data):
        if random.random() <= self.p:
            q_factor = random.uniform(self.q_factor[0], self.q_factor[1])
            encoding_param = [int(cv2.IMWRITE_JPEG_QUALITY), q_factor]

            img = data["image"]
            _, enc_img = cv2.imencode('.jpg', img, encoding_param)
            comp_img = cv2.imdecode(enc_img, 1)
            data["image"] = comp_img

        return data

    def __repr__(self):
        return '{}(q_factor=({}, {}), p={})'.format(self.__class__.__name__, self.q_factor[0], self.q_factor[1], self.p)


class RandomGaussianNoise(BaseTransformation):
    def __init__(self, opts):
        sigma_range = getattr(opts, "image_augmentation.random_gauss_noise.sigma", (0.03, 0.3))
        if isinstance(sigma_range, (float, int)):
            sigma_range = (0, sigma_range)

        assert len(sigma_range) == 2, 'Got {}'.format(sigma_range)
        assert sigma_range[0] <= sigma_range[1]
        p = getattr(opts, "image_augmentation.random_gauss_noise.p", 0.5)
        super(RandomGaussianNoise, self).__init__(opts=opts)
        self.sigma_low = sigma_range[0]
        self.sigma_high = sigma_range[1]
        self.p = p

    def __call__(self, data):
        if random.random() <= self.p:
            std = random.uniform(self.sigma_low, self.sigma_high)

            img = data["image"]
            noise = np.random.normal(0.0, std, img.shape) * 255
            noisy_img = img + noise

            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            data["image"] = noisy_img
        return data

    def __repr__(self):
        return '{}(sigma=({}, {}), p={})'.format(self.__class__.__name__, self.sigma_low, self.sigma_high, self.p)


class NumpyToTensor(BaseTransformation):
    def __init__(self, opts, *args, **kwargs):
        super(NumpyToTensor, self).__init__(opts=opts)

    def __call__(self, data):
        # HWC --> CHW
        img = data["image"]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # numpy to tensor
        img_tensor = torch.from_numpy(img).float()
        img_tensor = torch.div(img_tensor, 255.0)
        data["image"] = img_tensor

        mask = data.get("mask", None)
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).long()

        return data

class RandomOrder(BaseTransformation):
    def __init__(self, opts, img_transforms):
        super(RandomOrder, self).__init__(opts=opts)
        self.transforms = img_transforms
        apply_k_factor = getattr(opts, "image_augmentation.random_order.apply_k", 1.0)
        assert 0.0 < apply_k_factor <= 1.0, "--image-augmentation.random-order.apply-k should be > 0 and <= 1"
        self.keep_t = int(math.ceil(len(self.transforms) * apply_k_factor))

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.random-order.enable", action="store_true",
                           help="Ranoomly apply transforms")
        group.add_argument(
            "--image-augmentation.random-order.apply-k", type=int, default=1.0,
            help="Apply K percent of transforms randomly. Value between 0 and 1. Default is 1 (i.e., apply all)."
        )
        return parser

    def __call__(self, data):
        random.shuffle(self.transforms)
        for t in self.transforms[:self.keep_t]:
            data = t(data)
        return data

    def __repr__(self):
        transform_str = ", ".join(str(t) for t in self.transforms)
        repr_str = '{}(n_transforms={}, t_list=[{}]'.format(self.__class__.__name__, self.keep_t, transform_str)
        return repr_str


class Compose(BaseTransformation):
    def __init__(self, opts, img_transforms):
        super(Compose, self).__init__(opts=opts)
        self.img_transforms = img_transforms

    def __call__(self, data):
        for t in self.img_transforms:
            data = t(data)
        return data

    def __repr__(self):
        transform_str = ", ".join("\n\t\t\t" + str(t) for t in self.img_transforms)
        repr_str = '{}({})'.format(self.__class__.__name__, transform_str)
        return repr_str


class PhotometricDistort(BaseTransformation):
    def __init__(self, opts):
        beta_min = getattr(opts, "image_augmentation.photo_metric_distort.beta_min", -0.2)
        beta_max = getattr(opts, "image_augmentation.photo_metric_distort.beta_max", 0.2)
        assert -0.5 <= beta_min < beta_max <= 0.5, "Got {} and {}".format(beta_min, beta_max)

        alpha_min = getattr(opts, "image_augmentation.photo_metric_distort.alpha_min", 0.5)
        alpha_max = getattr(opts, "image_augmentation.photo_metric_distort.alpha_max", 1.5)
        assert 0 < alpha_min < alpha_max, "Got {} and {}".format(alpha_min, alpha_max)

        gamma_min = getattr(opts, "image_augmentation.photo_metric_distort.gamma_min", 0.5)
        gamma_max = getattr(opts, "image_augmentation.photo_metric_distort.gamma_max", 1.5)
        assert 0 < gamma_min < gamma_max, "Got {} and {}".format(gamma_min, gamma_max)

        delta_min = getattr(opts, "image_augmentation.photo_metric_distort.delta_min", -0.05)
        delta_max = getattr(opts, "image_augmentation.photo_metric_distort.delta_max", 0.05)
        assert -1.0 < delta_min < delta_max < 1.0, "Got {} and {}".format(delta_min, delta_max)

        super(PhotometricDistort, self).__init__(opts=opts)
        # for briightness
        self.beta_min = beta_min
        self.beta_max = beta_max
        # for contrast
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # for saturation
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        # for hue
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.p = getattr(opts, "image_augmentation.photo_metric_distort.p", 0.5)

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument("--image-augmentation.photo-metric-distort.enable", action="store_true",
                           help="Randomly apply photometric transformation")

        group.add_argument(
            "--image-augmentation.photo-metric-distort.alpha-min", type=float, default=0.5,
            help="Min. alpha value for contrast. Should be > 0"
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.alpha-max", type=float, default=1.5,
            help="Max. alpha value for contrast. Should be > 0"
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.beta-min", type=float, default=-0.2,
            help="Min. alpha value for brightness. Should be between -1 and 1."
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.beta-max", type=float, default=0.2,
            help="Max. alpha value for brightness. Should be between -1 and 1."
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.gamma-min", type=float, default=0.5,
            help="Min. alpha value for saturation. Should be > 0"
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.gamma-max", type=float, default=1.5,
            help="Max. alpha value for saturation. Should be > 0"
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.delta-min", type=float, default=-0.05,
            help="Min. alpha value for Hue. Should be between -1 and 1."
        )
        group.add_argument(
            "--image-augmentation.photo-metric-distort.delta-max", type=float, default=0.05,
            help="Max. alpha value for Hue. Should be between -1 and 1."
        )

        group.add_argument(
            "--image-augmentation.photo-metric-distort.p", type=float, default=0.5,
            help="Prob of applying transformation"
        )

        return parser

    def apply_transformations(self, image):

        def convert_to_uint8(img):
            return np.clip(img, 0, 255).astype(np.uint8)

        rand_nums = np.random.rand(6)

        image = image.astype(np.float32)

        # apply random contrast
        alpha = random.uniform(self.alpha_min, self.alpha_max) if rand_nums[0] < self.p else 1.0
        image *= alpha

        # Apply random brightness
        beta = (random.uniform(self.beta_min, self.beta_max) * 255) if rand_nums[1] < self.p else 0.0
        image += beta

        image = convert_to_uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float32)

        # Apply random saturation
        gamma = random.uniform(self.gamma_min, self.gamma_max) if rand_nums[2] < self.p else 1.0
        image[..., 1] *= gamma

        # Apply random hue
        delta = int(random.uniform(self.delta_min, self.delta_max) * 255) if rand_nums[3] < self.p else 0.0
        image[..., 0] += delta

        image = convert_to_uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if alpha == 1.0 and rand_nums[4] < self.p:
            # apply contrast if earlier not applied
            image = image.astype(np.float32)
            alpha = random.uniform(self.alpha_min, self.alpha_max)
            image *= alpha
            image = convert_to_uint8(image)

        # Lightning noise
        channels = image.shape[-1]
        swap = np.random.permutation(range(channels)) if rand_nums[5] < self.p else None
        if swap is not None:
            image = image[..., swap]

        return image

    def __call__(self, data):
        data["image"] = self.apply_transformations(data["image"])
        return data
