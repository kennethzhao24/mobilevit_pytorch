import torch
from .imagenet import ImagenetDataset


def build_eval_loader(args):
    eval_set = ImagenetDataset(args, is_training=False, is_evaluation=True)
    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=args.b, 
                                              shuffle=False,
                                              num_workers=args.workers, 
                                              pin_memory=True
                                              )

    return eval_loader
