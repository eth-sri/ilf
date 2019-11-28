import abc
from ilf.symbolic.symbolic import svm_utils
import z3
from enum import Enum
import logging
from ilf.symbolic.symbolic import constraints


CALLER_POOL_SIZE = 5
# SYMBOLIC_SHA_PADDING should be < SYMBOLIC_FIELD_PADDING
SYMBOLIC_SHA_PADDING = 1024
SYMBOLIC_FIELD_PADDING = 2048

class Storage(abc.ABC):

    @abc.abstractmethod
    def store(self, index, value):
        return


    @abc.abstractmethod
    def load(self, index):
        return

    @abc.abstractmethod
    def __copy__(self):
        return

    def load_part(self, index, start, stop):
        high = (stop+1) * 8 - 1
        low = start * 8
        full_data = self.load(index)
        return z3.Extract(high, low, full_data)

class EmptyStorage(Storage):
    def __init__(self):
        self.array = svm_utils.EMPTY_ARRAY

    def store(self, index, value):
        self.array = z3.Store(self.array, index, value)

    def load(self, index):
        return z3.Select(self.array, index)

    def __copy__(self):
        clone = EmptyStorage()
        clone.array = self.array
        return clone


class AbstractStorage(Storage):
    def __init__(self, label='abstract_storage'):
        self.array = z3.Array(label,
                                z3.BitVecSort(svm_utils.VECTOR_LEN),
                                z3.BitVecSort(svm_utils.VECTOR_LEN))

    def store(self, index, value):
        self.array = z3.Store(self.array, index, value)

    def load(self, index):
        return z3.Select(self.array, index)

    def __copy__(self):
        clone = AbstractStorage()
        clone.array = self.array
        return clone
