import random
from .base_sampler import BaseSamplerDDP, BaseSamplerDP


class BatchSampler(BaseSamplerDP):
    """
        Standard Batch Sampler for DP
    """
    def __init__(self, opts, n_data_samples, is_training = False):
        """
            :param opts: arguments
            :param n_data_samples: number of data samples in the dataset
            :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(BatchSampler, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w = getattr(opts, "sampler.bs.crop_size_width", 256)
        crop_size_h = getattr(opts, "sampler.bs.crop_size_height", 256)

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples:

            end_index = min(start_index + batch_size, self.n_samples)
            batch_ids = self.img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids]
                yield batch



class BatchSamplerDDP(BaseSamplerDDP):
    """
        Standard Batch Sampler for DDP
    """
    def __init__(self, opts, n_data_samples, is_training = False):
        """
            :param opts: arguments
            :param n_data_samples: number of data samples in the dataset
            :param is_training: Training or evaluation mode (eval mode includes validation mode)
        """
        super(BatchSamplerDDP, self).__init__(opts=opts, n_data_samples=n_data_samples, is_training=is_training)
        crop_size_w = getattr(opts, "sampler.bs.crop_size_width", 256)
        crop_size_h = getattr(opts, "sampler.bs.crop_size_height", 256)

        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h

    def __iter__(self):
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.img_indices)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):self.num_replicas]

        start_index = 0
        batch_size = self.batch_size_gpu0
        while start_index < self.n_samples_per_replica:
            end_index = min(start_index + batch_size, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != batch_size:
                batch_ids += indices_rank_i[:(batch_size - n_batch_samples)]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [(self.crop_size_h, self.crop_size_w, b_id) for b_id in batch_ids]
                yield batch
