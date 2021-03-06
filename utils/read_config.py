import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import yaml
import json
import collections.abc

__all__ = ['read_config']

def update(d, u):
    r""" deep update dict. copied from here: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def read_config(path_config: str, base=True):
    r""" read config yml file, return dict
    """
    new_config = yaml.safe_load(open(path_config))
    if not base:
        return new_config
    base_config = yaml.safe_load(open(new_config['base']))
    all_config = update(base_config, new_config)
    if all_config['lr_scheduler']['enable']:
        for key, value in all_config['lr_scheduler']['default'][all_config['lr_scheduler']['name']].items():
            if key not in all_config['lr_scheduler']:
                all_config['lr_scheduler'][key] = value
    return all_config
