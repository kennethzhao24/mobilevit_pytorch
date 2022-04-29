
import argparse
import time
import torch
import torch.backends.cudnn as cudnn

from common import model_cfg
from data import build_eval_loader
from models import create_mobilevit_model, model_cfg
from utils import AverageMeter, ProgressMeter, accuracy, load_pretrained

model_names = ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']

def get_eval_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation')
    parser.add_argument('--size', default=224, 
                        type=int, help='ImageNet resolustion')  
    parser.add_argument('--data', default='/common-data/ImageNet', 
                        type=str, help='ImageNet data path')                  
    parser.add_argument('--model', default='mobilevit_s',
                        choices=model_names, help='pytorch model')
    parser.add_argument('--workers', default=32, type=int,
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--b', default=100, type=int,
                        help='mini batch size')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    return parser


def main(args):
    # create model
    print("=> Creating model '{}'".format(args.model))
    if args.model.startswith('mobilevit'):
        model = create_mobilevit_model(model_cfg[args.model])
    else:
        raise ValueError('Model name invalid!')
    # load pretrained weights
    if args.pretrained:
        model = load_pretrained(args.pretrained, model)
        print('=> Pretrained Weighst Loaded!')
    # move model to gpu
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    # load imagenet data
    print("=> Loading imageNet data model")
    eval_loader = build_eval_loader(args)
    evaluate(eval_loader, model, args)


def evaluate(eval_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(eval_loader),
        [batch_time, top1, top5],
        prefix="Evaluate")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(eval_loader):
            images, target = batch['image'], batch['label']
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(True):
                output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


if __name__ == '__main__':
    parser = get_eval_arguments()
    args = parser.parse_args()
    main(args)
