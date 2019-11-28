import logging
import copy
import pprint
from collections import defaultdict

import z3

from ilf.symbolic.symbolic.machine_state import MachineState
from ilf.symbolic.utils import BColors
from ilf.symbolic import utils
from ilf.symbolic.symbolic import svm_utils
from copy import copy, deepcopy


class GlobalState:
    def __init__(self,
                 wstate,
                 environment,
                 mstate=None,
                 return_data=None,
                 pc_addr_to_depth=None,
                 has_storage_changed=False,
                 jump_count=0,
                 exit_code=None,
                 halt=False,
                 pc_trace=None,
                 loop_data=None
                 ):
        self.wstate = wstate
        self.wstate_idd = None
        self.environment = environment
        self.mstate = MachineState() if not mstate else mstate
        self.return_data = return_data
        self.pc_addr_to_depth = pc_addr_to_depth if pc_addr_to_depth else {}
        self.has_storage_changed = has_storage_changed
        self.jump_count = jump_count
        self.exit_code = exit_code
        self.halt = halt if halt is not None else False
        self.pc_trace = pc_trace if pc_trace is not None else []
        self.loop_data = loop_data if loop_data is not None else {}

    def __copy__(self):
        wstate = copy(self.wstate)
        environment = copy(self.environment)
        mstate = deepcopy(self.mstate)
        pc_addr_to_depth = copy(self.pc_addr_to_depth)
        return GlobalState(
            wstate,
            environment,
            mstate,
            self.return_data,
            pc_addr_to_depth,
            self.has_storage_changed,
            self.jump_count,
            self.exit_code,
            self.halt,
            copy(self.pc_trace),
            deepcopy(self.loop_data)
        )

    def __str__(self):
        return f'GlobalState: {id(self)}'

    def __repr__(self):
        return self.__str__()
