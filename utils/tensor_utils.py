import torch
from torch import distributed as dist


def tensor_size_from_opts(opts):
    try:
        sampler_name = getattr(opts, "sampler.name", "variable_batch_sampler").lower()
        if sampler_name.find("var") > -1:
            im_w = getattr(opts, "sampler.vbs.crop_size_width", 256)
            im_h = getattr(opts, "sampler.vbs.crop_size_height", 256)
        else:
            im_w = getattr(opts, "sampler.bs.crop_size_width", 256)
            im_h = getattr(opts, "sampler.bs.crop_size_height", 256)
    except Exception as e:
        im_w = im_h = 256
    return im_h, im_w


def create_rand_tensor(opts, device = "cpu"):
    im_h, im_w = tensor_size_from_opts(opts=opts)
    inp_tensor = torch.randint(low=0, high=255, size=(1, 3, im_h, im_w), device=device)
    inp_tensor = inp_tensor.float().div(255.0)
    return inp_tensor


def reduce_tensor(inp_tensor):
    size = float(dist.get_world_size())
    inp_tensor_clone = inp_tensor.clone()
    dist.barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def tensor_to_python_float(inp_tensor, is_distributed):
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        inp_tensor = reduce_tensor(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, 'item'):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError("The data type is not supported yet in tensor_to_python_float function")


def to_numpy(img_tensor):
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np
