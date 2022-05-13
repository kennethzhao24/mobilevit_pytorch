import numpy as np
import random
from utils.math_utils import make_divisible
from .base_sampler import BaseSamplerDP, BaseSamplerDDP


"""
    Varaible Batch Sampler for DP and DDP use cases. This sampler allows to sample different image sizes during training.
    For example, the first batch can contain 512 images of size 224x224, the second batch can contain 1024 imnages of 
    size 112x112, third batch can contain 384 images of size 224 x 112, and so on
    
    See paper for details: https://arxiv.org/abs/2110.02178?context=cs.LG
"""


def _image_batch_pairs(crop_size_w,
                       crop_size_h,
                       batch_size_gpu0,
                       n_gpus,
                       max_scales = 5,
                       check_scale_div_factor = 32,
                       min_crop_size_w = 160,
                       max_crop_size_w = 320,
                       min_crop_size_h = 160,
                       max_crop_size_h = 320,
                       *args, **kwargs):
    """
        This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    :param crop_size_w: Base Image width (e.g., 224)
    :param crop_size_h: Base Image height (e.g., 224)
    :param batch_size_gpu0: Batch size on GPU 0 for base image
    :param n_gpus: Number of available GPUs
    :param max_scales: Number of scales. How many image sizes that we want to generate between min and max scale factors.
    :param check_scale_div_factor: Check if image scales are divisible by this factor.
    :param min_crop_size_w: Min. crop size along width
    :param max_crop_size_w: Max. crop size along width
    :param min_crop_size_h: Min. crop size along height
    :param max_crop_size_h: Max. crop size along height
    :param args:
    :param kwargs:
    :return: a sorted list of tuples. Each index is of the form (h, w, batch_size)
    """

    width_dims = list(np.linspace(min_crop_size_w, max_crop_size_w, max_scales))
    if crop_size_w not in width_dims:
        width_dims.append(crop_size_w)

    height_dims = list(np.linspace(min_crop_size_h, max_crop_size_h, max_scales))
    if crop_size_h not in height_dims:
        height_dims.append(crop_size_h)

    image_scales = set()

    for h, w in zip(height_dims, width_dims):
        # ensure that sampled sizes are divisible by check_scale_div_factor
        # This is important in some cases where input undergoes a fixed number of down-sampling stages
        # for instance, in ImageNet training, CNNs usually have 5 downsampling stages, which downsamples the
        # input image of resolution 224x224 to 7x7 size
        h = make_divisible(h, check_scale_div_factor)
        w = make_divisible(w, check_scale_div_factor)
        image_scales.add((h, w))

    image_scales = list(image_scales)

    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for (crop_h, crop_y) in image_scales:
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(batch_size_gpu0, int(round(n_elements/(crop_h * crop_y), 2)))

        _bsz = make_divisible(_bsz, n_gpus)
        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


class VariableBatchSampler(BaseSamplerDP):
    """
        Variable batch sampler for DataParallel
    """
    def __init__(self, opts, n_data_samples, is_training = False):
        """
            :param opts: arguments
            :param n_data_samples: number of data samples in the dataset
            :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(VariableBatchSampler, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)

        crop_size_w = getattr(opts, "sampler.vbs.crop_size_width", 256)
        crop_size_h = getattr(opts, "sampler.vbs.crop_size_height", 256)

        min_crop_size_w = getattr(opts, "sampler.vbs.min_crop_size_width", 160)
        max_crop_size_w = getattr(opts, "sampler.vbs.max_crop_size_width", 320)

        min_crop_size_h = getattr(opts, "sampler.vbs.min_crop_size_height", 160)
        max_crop_size_h = getattr(opts, "sampler.vbs.max_crop_size_height", 320)

        scale_inc = getattr(opts, "sampler.vbs.scale_inc", False)
        scale_ep_intervals = getattr(opts, "sampler.vbs.ep_intervals", [40])
        scale_inc_factor= getattr(opts, "sampler.vbs.scale_inc_factor", 0.25)

        check_scale_div_factor = getattr(opts, "sampler.vbs.check_scale", 32)
        max_img_scales = getattr(opts, "sampler.vbs.max_n_scales", 10)

        if isinstance(scale_ep_intervals, int):
            scale_ep_intervals = [scale_ep_intervals]

        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

        self.scale_inc_factor = scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals

        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
        else:
            self.img_batch_tuples = [(crop_size_h, crop_size_w, self.batch_size_gpu0)]

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_tuples)

        start_index = 0
        while start_index < self.n_samples:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if len(batch_ids) != batch_size:
                batch_ids += self.img_indices[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        if epoch in self.scale_ep_intervals and self.scale_inc:
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.n_gpus,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
            if is_master_node:
                print('Scales updated in {}'.format(self.__class__.__name__))
                print("New scales: {}".format(self.img_batch_tuples))



class VariableBatchSamplerDDP(BaseSamplerDDP):
    """
        Variable batch sampler for DistributedDataParallel
    """
    def __init__(self, opts, n_data_samples, is_training = False):
        """
            :param opts: arguments
            :param n_data_samples: number of data samples in the dataset
            :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(VariableBatchSamplerDDP, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w = getattr(opts, "sampler.vbs.crop_size_width", 256)
        crop_size_h = getattr(opts, "sampler.vbs.crop_size_height", 256)

        min_crop_size_w = getattr(opts, "sampler.vbs.min_crop_size_width", 160)
        max_crop_size_w = getattr(opts, "sampler.vbs.max_crop_size_width", 320)

        min_crop_size_h = getattr(opts, "sampler.vbs.min_crop_size_height", 160)
        max_crop_size_h = getattr(opts, "sampler.vbs.max_crop_size_height", 320)

        scale_inc = getattr(opts, "sampler.vbs.scale_inc", False)
        scale_ep_intervals = getattr(opts, "sampler.vbs.ep_intervals", [40])
        scale_inc_factor = getattr(opts, "sampler.vbs.scale_inc_factor", 0.25)
        check_scale_div_factor = getattr(opts, "sampler.vbs.check_scale", 32)

        max_img_scales = getattr(opts, "sampler.vbs.max_n_scales", 10)

        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.min_crop_size_h = min_crop_size_h
        self.max_crop_size_h = max_crop_size_h
        self.min_crop_size_w = min_crop_size_w
        self.max_crop_size_w = max_crop_size_w

        self.scale_inc_factor = scale_inc_factor
        self.scale_ep_intervals = scale_ep_intervals
        self.max_img_scales = max_img_scales
        self.check_scale_div_factor = check_scale_div_factor
        self.scale_inc = scale_inc

        if is_training:
            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
        else:
            self.img_batch_tuples = [(self.crop_size_h, self.crop_size_w, self.batch_size_gpu0)]

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_tuples)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]

        start_index = 0
        while start_index < self.n_samples_per_replica:
            crop_h, crop_w, batch_size = random.choice(self.img_batch_tuples)

            end_index = min(start_index + batch_size, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(crop_h, crop_w, b_id) for b_id in batch_ids]
                yield batch

    def update_scales(self, epoch, is_master_node=False, *args, **kwargs):
        if (epoch in self.scale_ep_intervals) and self.scale_inc:  # Training mode
            self.min_crop_size_w += int(self.min_crop_size_w * self.scale_inc_factor)
            self.max_crop_size_w += int(self.max_crop_size_w * self.scale_inc_factor)

            self.min_crop_size_h += int(self.min_crop_size_h * self.scale_inc_factor)
            self.max_crop_size_h += int(self.max_crop_size_h * self.scale_inc_factor)

            self.img_batch_tuples = _image_batch_pairs(
                crop_size_h=self.crop_size_h,
                crop_size_w=self.crop_size_w,
                batch_size_gpu0=self.batch_size_gpu0,
                n_gpus=self.num_replicas,
                max_scales=self.max_img_scales,
                check_scale_div_factor=self.check_scale_div_factor,
                min_crop_size_w=self.min_crop_size_w,
                max_crop_size_w=self.max_crop_size_w,
                min_crop_size_h=self.min_crop_size_h,
                max_crop_size_h=self.max_crop_size_h
            )
            if is_master_node:
                print('Scales updated in {}'.format(self.__class__.__name__))
                print('Min. scale: {}->{}, Max.scale: {}->{}'.format(self.min_scale - self.scale_inc_factor,
                                                                          self.min_scale,
                                                                          self.max_scale - self.scale_inc_factor,
                                                                          self.max_scale))
                print("New scales: {}".format(self.img_batch_tuples))

