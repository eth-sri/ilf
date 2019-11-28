from ilf.symbolic.symbolic import svm_utils
from copy import copy, deepcopy


class MachineState:
    def __init__(
            self,
            pc=0,
            stack=None,
            memory=None,
            memory_dict=None,
            gas=10**6):
        self.pc = pc if pc else 0
        self.stack = stack if stack else []
        self.memory = memory if memory is not None else svm_utils.get_zero_array(256, 8)
        self.memory_dict = memory_dict if memory_dict else {}
        self.gas = gas

    def __str__(self):
        return str({'pc': self.pc, 'stack': self.stack, 'memory': self.memory, 'gas': self.gas})

    def __deepcopy__(self, memo):
        return MachineState(
                self.pc,
                copy(self.stack),
                self.memory,
                copy(self.memory_dict),
                self.gas)
