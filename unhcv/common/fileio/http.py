import wget
import os

def download_from_http(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    wget.download(url, dest)