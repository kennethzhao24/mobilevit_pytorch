
import numpy as np
import random
import socket
import torch
import torch.distributed as dist

def is_master(opts):
    node_rank = getattr(opts, "ddp.rank", 0)
    return (node_rank == 0)

def device_setup(opts):
    random_seed = getattr(opts, "seed", 0)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    is_master_node = is_master(opts)
    if is_master_node:
        print('Random seeds are set to {}'.format(random_seed))
        print('Using PyTorch version {}'.format(torch.__version__))

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        if is_master_node:
            print('No GPUs available. Using CPU')
        device = torch.device('cpu')
        n_gpus = 0
    else:
        if is_master_node:
            print('Available GPUs: {}'.format(n_gpus))
        device = torch.device('cuda')

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            torch.backends.cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            if is_master_node:
                print('CUDNN is enabled')

    setattr(opts, "dev.device", device)
    setattr(opts, "dev.num_gpus", n_gpus)

    return opts

    ddp_url = getattr(opts, "ddp.dist_url", None)
    ddp_port = getattr(opts, "ddp.dist_port", 6006)
    is_master_node = is_master(opts)
    if ddp_url is None:
        hostname = socket.gethostname()
        ddp_url = 'tcp://{}:{}'.format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)
    if torch.distributed.is_initialized():
        print('DDP is already initialized and cannot be initialize twice!')
    else:
        print('distributed init (rank {}): {}'.format(node_rank, ddp_url))

        dist_backend = "gloo"
        if dist.is_nccl_available():
            dist_backend = 'nccl'
            if is_master_node:
                print('Using NCCL as distributed backend with version={}'.format(torch.cuda.nccl.version()))

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank