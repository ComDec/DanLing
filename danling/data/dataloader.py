import torch
from torch.utils import data

from .utils import send_to_device


class DataLoader(data.DataLoader):
    def __init__(self, *args, device: torch.device | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __iter__(self):
        return DataLoaderIterWrapper(super().__iter__(), self.device)


class DataLoaderIterWrapper:
    def __init__(self, dataloader_iter: data.dataloader._BaseDataLoaderIter, device: torch.device | None = None):
        self.dataloader_iter = dataloader_iter
        self.device = device

    def __iter__(self):
        return iter(self.dataloader_iter)

    def __next__(self):
        return send_to_device(next(self.dataloader_iter), self.device) if self.device else next(self.dataloader_iter)
