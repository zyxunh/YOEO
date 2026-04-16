import os
import subprocess
import shutil
import argparse
from typing import List
from unhcv.common.array import chunk
from unhcv.common.utils import obj_load, remove_dir

def listdir(path: str, kv=False) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if path.startswith('hdfs://'):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        try:
            files = [os.path.join(path, file) for file in os.listdir(path)]
        except:
            files = []
    if kv:
        files = [os.path.splitext(var)[0] for var in files if var.endswith(".index")]

    return files

def mkdir(path: str):
    """
    Create directory. Support either hdfs or local path.
    Create all parent directory if not present. No-op if directory already present.
    """
    if path.startswith('hdfs://'):
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str, asynchronization=False, exist_ok=True):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    if not exist_ok:
        assert not exists(tgt)
    if os.path.basename(src) == os.path.basename(tgt):
        tgt = os.path.dirname(tgt)
    if not exists(tgt):
        mkdir(tgt)
    src_hdfs = src.startswith("hdfs://")
    tgt_hdfs = tgt.startswith("hdfs://")

    if asynchronization:
        copy_func = subprocess.Popen
    else:
        copy_func = subprocess.run
    if src_hdfs and tgt_hdfs:
        return copy_func(["hdfs", "dfs", "-cp", "-f", src, tgt])
    elif src_hdfs and not tgt_hdfs:
        return copy_func(["hdfs", "dfs", "-copyToLocal", "-f", src, tgt])
    elif not src_hdfs and tgt_hdfs:
        return copy_func(["hdfs", "dfs", "-copyFromLocal", "-f", src, tgt])
    else:
        shutil.copy(src, tgt)


def exists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("hdfs dfs -test -e {}".format(file_path)) == 0
    return os.path.exists(file_path)


def hdfs_io(file_paths, save_paths, max_procs_num=10, mode="download"):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if isinstance(save_paths, str):
        save_paths = [save_paths] * len(file_paths)
    if len(file_paths) > max_procs_num:
        _file_paths = chunk(file_paths, max_procs_num)
        _save_paths = chunk(save_paths, max_procs_num)
    else:
        _file_paths = [file_paths]
        _save_paths = [save_paths]
    for file_paths, save_paths in zip(_file_paths, _save_paths):
        procs = []
        for file_path, save_path in zip(file_paths, save_paths):
            if mode == "download":
                os.makedirs(save_path, exist_ok=True)
                print(f'download {file_path} to {save_path}')
                # breakpoint()
                if file_path.startswith('hdfs'):
                    proc = subprocess.Popen(['hdfs', 'dfs', '-get', file_path, save_path])
                else:
                    proc = subprocess.Popen(['cp', '-r', file_path, save_path])
            elif mode == "upload":
                assert save_path.startswith('hdfs')
                stdout, stderr = subprocess.Popen(['hdfs', 'dfs', '-mkdir', "-p", save_path]).communicate()
                # print(f'mkdiring, stdout is {stdout}, strderr is {stderr}')
                print(f'upload {file_path} to {save_path}')
                proc = subprocess.Popen(['hdfs', 'dfs', '-put', file_path, save_path])
            procs.append(proc)
        for proc in procs:
            stdout, stderr = proc.communicate()
            if stdout is not None or stderr is not None:
                print(f'stdout is {stdout}, strderr is {stderr}')

def get_parser():
    parser = argparse.ArgumentParser(
        description='hdfs tools')
    parser.add_argument(
        '--data_list_file', type=str, default=None)
    parser.add_argument(
        '--mode', type=str, default='download')
    parser.add_argument(
        '--file_paths', type=str, nargs='+', default=None)
    parser.add_argument(
        '--save_paths', type=str, nargs='+', default=None)
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_parser()
    if args.data_list_file is not None:
        data_list = obj_load(args.data_list_file)
    elif args.file_paths is not None:
        file_paths = args.file_paths
        save_paths = args.save_paths
        if len(file_paths) == 1:
            file_paths = file_paths[0]
        if len(save_paths) == 1:
            save_paths = save_paths[0]
        data_list = [dict(file_paths=file_paths, save_paths=save_paths)]
    for data in data_list:
        if isinstance(data['file_paths'], str):
            data['file_paths'] = [data['file_paths']]
        for _file_path in data['file_paths']:
            file_paths = listdir(_file_path)
            save_paths = data['save_paths']
            if file_paths[0] != _file_path:
                save_paths = os.path.join(save_paths, os.path.basename(_file_path))
            hdfs_io(file_paths, save_paths, mode=args.mode)
