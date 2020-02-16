import torch.utils.data as data


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
