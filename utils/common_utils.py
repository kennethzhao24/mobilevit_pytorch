import collections
import yaml

def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file(opts):
    config_file_name = getattr(opts, "config", None)
    with open(config_file_name, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

        flat_cfg = flatten_yaml_as_dict(cfg)
        for k, v in flat_cfg.items():
            setattr(opts, k, v)
    return opts
