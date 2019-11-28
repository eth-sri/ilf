from copy import copy, deepcopy
from ilf.symbolic import utils
from collections import defaultdict
from enum import Enum
from ilf.symbolic.symbolic import svm_utils
import z3
from ilf.symbolic.utils import BColors

class WorldStateStatus(Enum):
    UNKNOWN = 1
    FEASIBLE = 2
    INFEASIBLE = 3

class WorldState:
    def __init__(self,
                 constraints=None,
                 address_to_account=None,
                 children=None,
                 gen=None,
                 trace=None,
                 parent=None,
                 storage_writes=None,
                 prev_ref_to_eval=None,
                 has_symbolic_calls=False
                 ):
        self.constraints = constraints if constraints else []
        self.address_to_account = address_to_account if address_to_account else {}
        self.children = children if children else []
        self.parent = parent
        self.gen = gen if gen else 0
        self.trace = trace
        self.status = WorldStateStatus.UNKNOWN
        self.storage_writes = storage_writes if storage_writes is not None else []
        self.prev_ref_to_eval = prev_ref_to_eval if prev_ref_to_eval is not None else {}
        self.has_symbolic_calls = has_symbolic_calls if has_symbolic_calls is not None else False


    def __copy__(self):
        address_to_account = deepcopy(self.address_to_account)
        return WorldState(
                copy(self.constraints),
                address_to_account,
                copy(self.children),
                self.gen,
                self.trace,
                self.parent,
                copy(self.storage_writes),
                copy(self.prev_ref_to_eval),
                self.has_symbolic_calls
                )

    def child(self):
        child_wstate = copy(self)
        child_wstate.gen += 1
        child_wstate.parent = self
        child_wstate.storage_writes = []
        child_wstate.has_symbolic_calls = False
        child_wstate.trace = None
        return child_wstate


    def abstract(self):
        for address, account in self.address_to_account.items():
            label_suffix = '_{}'.format(address)
            account.abstract(label_suffix)
        self.non_abstracted_constraints = self.constraints
        self.constraints = []
        self.predicate_to_constraints = {}
        old_prev_values = self.prev_ref_to_eval
        self.prev_ref_to_eval = {}
        for ref in old_prev_values:
            label = f'prev_value_id{hash(ref)}_gen{self.gen}'
            self.prev_ref_to_eval[ref] = z3.BitVec(label, 256)

    def __str__(self):
        return f'{BColors.GREEN}{BColors.BOLD}World State:{BColors.ENDC} {id(self)}. Gen:{self.gen}, trace: {self.trace}'

    def __repr__(self):
        return f'wstate_{id(self)}'

    def get_full_trace(self):
        full_trace = []
        full_trace.append(self.trace)
        parent_wstate = self.parent
        while parent_wstate is not None:
            full_trace.append(parent_wstate.trace)
            parent_wstate = parent_wstate.parent
        return full_trace[::-1]
