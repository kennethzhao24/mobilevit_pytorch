import argparse
import time
import torch

from dataset import build_dataloader
from lib.models import build_model
from utils.common_utils import load_config_file
from utils.ddp_utils import is_master, device_setup
from utils.metrics import metric_monitor
from utils.model_utils import load_pretrained_model
from utils.stats import Statistics



def get_eval_arguments(parse_args = True):
    parser = argparse.ArgumentParser(description='ImageNet Evaluation', add_help=True)
    parser.add_argument('--config', type=str, default='./configs/imagenet.yaml', help="Configuration file")
    parser.add_argument('--model_name', type=str, default='mobilevit_xs', help="Model name")
    parser.add_argument('--weights', type=str, default='.')

    if parse_args:
        # parse args
        opts = parser.parse_args()
        opts = load_config_file(opts)
        return opts
    else:
        return parser
    

def main():
    opts = get_eval_arguments()
    # device set-up
    opts = device_setup(opts)
    device = getattr(opts, "dev.device", torch.device('cpu'))
    is_master_node = is_master(opts)

    # load imagenet data
    eval_loader = build_dataloader(opts)

    # set-up the model
    model = build_model(opts)

    # load pretrained weights
    weights = getattr(opts, "weights", None)
    if weights is not None:
        model = load_pretrained_model(model=model, wt_loc=weights, is_master_node=is_master_node)
    else:
        raise ValueError('No model weights found!')
    
    # move model to device
    model = model.to(device=device)

    evalute_engine(opts, model, eval_loader, device)


def evalute_engine(opts, model, eval_loader, device):
    is_master_node = is_master(opts)
    evaluation_stats = Statistics(metric_names=['top1', 'top5'], 
                                  is_master_node=is_master_node
                                  )
    model.eval()
    eval_start_time = time.time()
    with torch.no_grad():
        epoch_start_time = time.time()
        total_samples = len(eval_loader)
        processed_samples = 0

        for batch_id, batch in enumerate(eval_loader):
            input_img, target_label = batch['image'], batch['label']

            # move data to device
            input_img = input_img.to(device)
            target_label = target_label.to(device)
            batch_size = input_img.shape[0]

            # prediction
            pred_label = model(input_img)

            processed_samples += batch_size
            metrics = metric_monitor(pred_label=pred_label, target_label=target_label, 
                                     loss=0.0, metric_names=['top1', 'top5']
                                     )

            evaluation_stats.update(metric_vals=metrics, batch_time=0.0, n=batch_size)

            if batch_id % batch_size == 0 and is_master_node:
                evaluation_stats.eval_summary(n_processed_samples=processed_samples,
                                              total_samples=total_samples,
                                              elapsed_time=epoch_start_time)
                                              
    evaluation_stats.epoch_summary(stage="evaluation")
    eval_end_time = time.time() - eval_start_time
    print('Evaluation took {:.2f} seconds'.format(eval_end_time))


if __name__ == "__main__":
    main()
