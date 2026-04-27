import os
import yaml
from urllib.parse import quote


class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)


def register_yaml_constructors():
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)


def conf_edit(config_path, chunk_size, overlap):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True
    # ensure chunk_size is an int (users may supply strings from Colab widgets)
    try:
        cs = int(chunk_size)
    except Exception:
        try:
            cs = int(float(chunk_size))
        except Exception:
            cs = chunk_size
    data['audio']['chunk_size'] = cs
    data['inference']['num_overlap'] = overlap
    if data['inference']['batch_size'] == 1:
        data['inference']['batch_size'] = 2
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)


def download_file(url):
    encoded_url = quote(url, safe=':/')
    path = 'ckpts'
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        return
    try:
        try:
            import torch
            torch.hub.download_url_to_file(encoded_url, file_path)
        except Exception:
            import urllib.request
            urllib.request.urlretrieve(encoded_url, file_path)
    except Exception:
        pass
