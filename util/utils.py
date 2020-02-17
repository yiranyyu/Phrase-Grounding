import random
import subprocess

import numpy as np
import torch


def cuda_is_available():
    import ctypes
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            del cuda
            try:
                import torch.cuda
            except ImportError:
                continue
            return True
    else:
        return False


def cuda_device_count(python_interpreter='python'):
    import os
    key = 'CUDA_VISIBLE_DEVICES'
    if key in os.environ:
        return len(os.environ[key].split(','))
    else:
        return int(subprocess.getoutput(f"{python_interpreter} -c 'import torch as th; print(th.cuda.device_count())'"))


def set_random_seed(s, deterministic=True):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

    if cuda_is_available():
        torch.cuda.manual_seed_all(s)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = not deterministic
        cudnn.deterministic = deterministic
