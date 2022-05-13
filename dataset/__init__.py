from torch.utils.data.dataloader import DataLoader
from .imagenet import ImagenetDataset
from .sampler import build_sampler

def build_dataloader(opts):
    dataset_name = getattr(opts, "dataset.name", 'imagenet')
    if dataset_name == 'imagenet':
        eval_set = ImagenetDataset(opts=opts, is_training=False, is_evaluation=True) 

    setattr(opts, "dataset.val_batch_size", getattr(opts, "dataset.eval_batch_size", 1))
    eval_sampler = build_sampler(opts=opts, n_data_samples=len(eval_set), is_training=False)

    data_workers = getattr(opts, "dataset.workers", 1)
    persistent_workers = False
    pin_memory = False

    eval_loader = DataLoader(dataset=eval_set,
                                batch_size=1,
                                batch_sampler=eval_sampler,
                                num_workers=data_workers,
                                pin_memory=pin_memory,
                                persistent_workers=persistent_workers
                                )
        
    return eval_loader
