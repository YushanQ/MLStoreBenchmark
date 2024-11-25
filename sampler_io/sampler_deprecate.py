import os
import itertools
import torch
import numpy as np
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

class Sampler:
    """Base sampler class"""
    def __init__(self, data_source: Optional[Sized] = None):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data_source)

class SequentialSampler(Sampler):
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

class RandomSampler(Sampler):
    """Bdefault config of pytorch
    replacement = false
    generator = default
    sample = len(data source)
    """
    def __iter__(self) -> Iterator[int]:
        indices =  np.random.permutation(len(self.data_source))
        return iter(indices)


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class SimpleDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle: bool = False,
            drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        # Set up sampling strategy
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        if batch_size is not None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self) -> "_BaseDataLoaderIter":
        return self._get_iterator()

    # def _get_iterator(self) -> "_BaseDataLoaderIter":
    #     if self.num_workers == 0:
    #         return _SingleProcessDataLoaderIter(self)

    def __len__(self):
        length = len(self.dataset)
        if (self.batch_size is not None):
            from math import ceil

            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length

# class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
#     def __init__(self, loader):
#         super().__init__(loader)
#         assert self._timeout == 0
#         assert self._num_workers == 0
#
#         # Adds forward compatibilities so classic DataLoader can work with DataPipes:
#         #   Taking care of distributed sharding
#         if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
#             # For BC, use default SHARDING_PRIORITIES
#             torch.utils.data.graph_settings.apply_sharding(
#                 self._dataset, self._world_size, self._rank
#             )
#
#         self._dataset_fetcher = _DatasetKind.create_fetcher(
#             self._dataset_kind,
#             self._dataset,
#             self._auto_collation,
#             self._collate_fn,
#             self._drop_last,
#         )
#
#     def _next_data(self):
#         index = self._next_index()  # may raise StopIteration
#         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
#         if self._pin_memory:
#             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
#         return data

if __name__ == "__main__":
    # do nothing
    print("do nothing")
