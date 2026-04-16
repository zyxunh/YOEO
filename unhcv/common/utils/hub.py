from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
import tarfile
from zipfile import ZipFile
import os.path as osp
import os
# import wget

import torch

def download(url, dir, save_name=None, unzip=False, delete=False):
    if save_name is None:
        os.makedirs(dir, exist_ok=True)
        if isinstance(dir, str):
            dir = Path(dir)
        f = dir / Path(url).name
    else:
        save_name = osp.join(dir, save_name)
        f = Path(save_name)
        dir =  osp.splitext(save_name)[0]
        os.makedirs(dir, exist_ok=True)
        print('dir is ', dir)
    if Path(url).is_file():
        Path(url).rename(f)
    elif not f.exists():
        print(f'Downloading {url} to {f}')
        # os.system(f'export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org && \
        #           wget -c -t 100 {url} -O {f}')
        # wget.download(f'{url}', f'{f}')
        torch.hub.download_url_to_file(url, f, progress=False)
    else:
        pass
    if unzip and f.suffix in ('.zip', '.tar', '.gz'):
        print(f'Unzipping {f.name}')
        if f.suffix == '.zip':
            ZipFile(f).extractall(path=dir)
        elif f.suffix in ['.tar', '.gz']:
            with tarfile.open(str(f), 'r') as tf:
                flag = tf.extractall(f'{dir}')
        if delete:
            f.unlink()
            print(f'Delete {f}')
    return f

def download_(url, dir, save_name=None, unzip=True, delete=False, threads=1):
    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        if save_name is None:
            save_name = repeat(save_name)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir), save_name))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir, save_name)
