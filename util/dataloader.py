import collections
import math
import re
import traceback

import torch
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler

_use_shared_memory = False

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class SliceDataset(data.Dataset):
    def __init__(self, src, offset=0, size=0):
        offset = offset if offset >= 0 else len(src) + offset
        size = len(src) - offset if size == 0 else size
        assert (0 <= offset < len(src))
        assert (0 <= size <= len(src) - offset)
        self.src = src
        self.offset = offset
        self.size = size

    def __getitem__(self, index):
        index = index if index >= 0 else self.size + index
        if index < self.size:
            return self.src[self.offset + index]
        raise IndexError(f'index {index} out of range {self.size}')

    def __len__(self):
        return self.size


class ExceptionWrapper(object):
    """Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def default_collate(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
        Args:
            batch: list of tuples of (inputs, targets) => [inputs=[ctx0,ctx1,...], targets]
    """

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader that combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    NOTE: Specify nbatches to iterate only a subset of the entire dataset one time.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        nbatches (int, optional): how many batches to iterate in one epoch
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, nbatches=0, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):

        super(DataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                         pin_memory, drop_last)

        # XXX custom
        self.src = dataset
        self.nbatches = min(nbatches, int(math.ceil(len(dataset) / batch_size))) if nbatches > 0 else int(
            math.ceil(len(dataset) / batch_size))
        self.shuffle = shuffle
        self.offset = 0

    def __iter__(self):
        # XXX Each time a different subset of examples are iterated in nbatches.

        sampler = self.sampler
        batch_sampler = self.batch_sampler

        size = self.batch_size * self.nbatches if self.drop_last else min(self.batch_size * self.nbatches,
                                                                          len(self.src) - self.offset)
        self.offset = self.offset if size <= len(self.src) - self.offset else 0
        self.dataset = SliceDataset(self.src, offset=self.offset, size=size)
        if batch_sampler is None:
            if sampler is None:
                if self.shuffle:
                    sampler = RandomSampler(self.dataset)
                else:
                    sampler = SequentialSampler(self.dataset)
            batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)

        self.sampler, sampler = sampler, self.sampler
        self.batch_sampler, batch_sampler = batch_sampler, self.batch_sampler
        itr = super(DataLoader, self).__iter__()

        self.offset = (self.offset + len(self.dataset)) % len(self.src)
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        return itr

    def __len__(self):
        return self.batch_sampler and len(self.batch_sampler) or (
                len(self.dataset) + self.batch_size - 1) // self.batch_size
