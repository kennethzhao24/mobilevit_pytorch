from utils.tensor_utils import tensor_to_python_float


def top_k_accuracy(output, target, top_k = (1,)):
    maximum_k = max(top_k)
    batch_size = target.shape[0]

    _, pred = output.topk(maximum_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(
        target.reshape(1, -1).expand_as(pred)
    )

    results = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_k = correct_k.mul_(100.0 / batch_size)
        results.append(acc_k)
    return results


def metric_monitor(pred_label, target_label, loss, metric_names, use_distributed = False):
    metric_vals = dict()
    if "loss" in metric_names:
        loss = tensor_to_python_float(loss, is_distributed=use_distributed)
        metric_vals['loss'] = loss

    if "top1" in metric_names:
        top_1_acc, top_5_acc = top_k_accuracy(pred_label, target_label, top_k=(1, 5))
        top_1_acc = tensor_to_python_float(top_1_acc, is_distributed=use_distributed)
        metric_vals['top1'] = top_1_acc
        if "top5" in metric_names:
            top_5_acc = tensor_to_python_float(top_5_acc, is_distributed=use_distributed)
            metric_vals['top5'] = top_5_acc

    return metric_vals