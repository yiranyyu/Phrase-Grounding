from torch import nn
import torch as th

from util import logging


class DataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, attr):
        if attr == "module":
            return self._modules["module"]
        else:
            val = getattr(self.module, attr)
            return callable(val) and val or val

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Override to avoid saving DataParallel nonsense.
        """
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """Override to avoid loading DataParallel nonsense.
        """
        self.module.load_state_dict(state_dict, strict)


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        """TODO: Testing
        """
        super(DistributedDataParallel, self).__init__(module, device_ids, output_device, dim)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict)


def parallelize(module, device_ids=None, distributed=False):
    """
    Args:
        device_ids:
            None: all available GPUs
            str: number or comma separated list
            int: number of GPUs to parallelize
            list: int GPU device ids
    """

    available = list(range(th.cuda.device_count()))
    if device_ids is None:
        device_ids = available
    elif isinstance(device_ids, str):
        # number or comma separated list
        device_ids = (
            available[-int(device_ids):]
            if device_ids.isnumeric()
            else [int(i) for i in device_ids.split(",")]
        )
    assert all(
        i in available for i in device_ids
    ), f"Not all requested GPUs {device_ids} are available in {available}"

    device = th.device(f"cuda:{device_ids[0]}") if device_ids else th.device("cpu")
    module.to(device)
    if distributed:
        # TODO: testing
        return DistributedDataParallel(module)
    elif len(device_ids) > 1:
        module = nn.DataParallel(module, device_ids)
        logging.info(f"Parallelized module on GPUs: {device_ids}")

    return module, device
