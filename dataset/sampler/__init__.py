from .batch_sampler import BatchSampler, BatchSamplerDDP
from .variable_batch_sampler import VariableBatchSampler, VariableBatchSamplerDDP


def build_sampler(opts, n_data_samples, is_training = False):
    sampler_name = getattr(opts, "sampler.name", "variable_batch_sampler")
    is_distributed = getattr(opts, "ddp.use_distributed", False)
    sampler = None
    if sampler_name == "variable_batch_sampler":
        if is_distributed:
            sampler = VariableBatchSamplerDDP(opts, n_data_samples=n_data_samples, is_training=is_training)
        sampler = VariableBatchSampler(opts, n_data_samples=n_data_samples, is_training=is_training)
    if sampler_name == "batch_sampler":
        if is_distributed:
            sampler = BatchSamplerDDP(opts, n_data_samples=n_data_samples, is_training=is_training)
        sampler = BatchSampler(opts, n_data_samples=n_data_samples, is_training=is_training)

    return sampler

