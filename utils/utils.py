import errno
import os

import torch


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def training_device(device='cuda'):
    return 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'