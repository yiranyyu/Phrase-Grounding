import torch as th

from torch import nn
from util import logging


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__(module, device_ids, output_device, dim)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict)


def parallelize(module, device_ids=None, distributed=False):
    device = th.device(f"cuda:0") if device_ids else th.device("cpu")
    module.to(device)
    if distributed:
        # TODO: testing
        return DistributedDataParallel(module)
    elif len(device_ids) > 1:
        logging.info(f"Paralleled module on GPUs: {device_ids}")
        module = nn.DataParallel(module)

    return module, device
