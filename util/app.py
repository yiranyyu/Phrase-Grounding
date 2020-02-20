import os
import sys

from util import logging
from util.utils import cuda_device_count


def init_cuda(cfg):
    if cfg.no_gpu:
        cfg.gpu = []
        os.environ['CUDA_VISIBLE_DEVICES'] = 'NoDevFiles'
    elif cfg.gpu is None:
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] != 'NoDevFiles':
            cfg.gpu = sorted(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        else:
            cfg.gpu = list(range(cuda_device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(map(str, cfg.gpu)))


def init(cfg):
    init_cuda(cfg)
    logging.info(f"Logging to {cfg.logfile}")
    handlers = [logging.FileHandler(cfg.logfile)]
    if cfg.log_std:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(handlers=handlers)


def execute(main, cfg, *args, **kwargs):
    init(cfg)
    main(cfg, *args, **kwargs)


def run(main, cfg, *args, **kwargs):
    # Local worker with cfg.gpu
    execute(main, cfg, *args, **kwargs)
