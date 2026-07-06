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


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def conf_edit(config_path, chunk_size, overlap):
    data = load_yaml_config(config_path)
    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True
    # chunk_size None means "keep the chunk_size from the model's own yaml"
    if chunk_size is not None:
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


def download_file(url, dest_path=None):
    """Download url to dest_path (atomic). When dest_path is omitted, saves to
    ckpts/<url basename> for backwards compatibility."""
    encoded_url = quote(url, safe=':/')
    if dest_path is None:
        dest_path = os.path.join('ckpts', os.path.basename(encoded_url))
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path
    tmp_path = dest_path + '.part'
    try:
        try:
            import torch
            torch.hub.download_url_to_file(encoded_url, tmp_path)
        except Exception:
            import urllib.request
            urllib.request.urlretrieve(encoded_url, tmp_path)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception as e:
        print(f'Download failed for {url}: {e}')
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None
