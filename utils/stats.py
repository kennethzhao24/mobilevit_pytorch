import sys
import time

SUPPORTED_STATS = [
    'loss',
    'top1', 
    'top5',
]

class Statistics(object):
    def __init__(self, 
                 metric_names = ['loss'], 
                 is_master_node = False
                 ):
        if len(metric_names) == 0:
            raise ValueError('Metric names list cannot be empty')

        # key is the metric name and value is the value
        metric_dict  = {}
        metric_counters = {}
        for m_name in metric_names:
            if m_name in SUPPORTED_STATS:
                metric_dict[m_name] = None
                metric_counters[m_name] = 0
            else:
                if is_master_node:
                    print('{} statistics not supported. Supported: {}'.format(m_name, SUPPORTED_STATS))

        self.metric_dict = metric_dict
        self.supported_metrics = list(metric_dict.keys())
        self.metric_counters = metric_counters
        self.round_places = 4
        self.is_master_node = is_master_node

        self.batch_time = 0
        self.batch_counter = 0

    def update(self, metric_vals, batch_time, n = 1):
        for k, v in metric_vals.items():
            if k in self.supported_metrics:
                if self.metric_dict[k] is None:
                    self.metric_dict[k] = v * n
                else:
                    self.metric_dict[k] += v * n
                self.metric_counters[k] += n
        self.batch_time += batch_time
        self.batch_counter += 1

    def avg_statistics_all(self, sep=": "):
        metric_stats = []
        for k, v in self.metric_dict.items():
            counter = self.metric_counters[k]
            v_avg = (v * 1.0) / counter
            v_avg = round(v_avg, self.round_places)
            metric_stats.append("{:<}{}{:.4f}".format(k, sep, v_avg))
        return metric_stats

    def avg_statistics(self, metric_name):
        avg_val = None
        if metric_name in self.supported_metrics:
            counter = self.metric_counters[metric_name]
            v = self.metric_dict[metric_name]
            avg_val = (v * 1.0) / counter
            avg_val = round(avg_val, self.round_places)
        return avg_val

    def eval_summary(self, n_processed_samples, total_samples, elapsed_time):
        if self.is_master_node:
            metric_stats = self.avg_statistics_all()
            el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)
            epoch_str = "Evaluation:[{:8d}/{:8d}]".format(n_processed_samples, total_samples)
            batch_str = "Avg. batch load time: {:1.3f}".format(self.batch_time / self.batch_counter)
            stats_summary = [epoch_str]
            stats_summary.extend(metric_stats)
            stats_summary.append(batch_str)
            stats_summary.append(el_time_str)
            summary_str = ", ".join(stats_summary)
            print(summary_str)
            sys.stdout.flush()

    def epoch_summary(self, stage = "Training"):
        if self.is_master_node:
            metric_stats = self.avg_statistics_all(sep="=")
            metric_stats_str = " || ".join(metric_stats)
            print('*** {} summary:'.format(stage.title()))
            print("\t {}".format(metric_stats_str))
            sys.stdout.flush()
