import abc
import math
import json
from collections import defaultdict
from collections import OrderedDict
import random
from ...execution import Tx
import z3
from ..policy_base import PolicyBase
from ...ethereum import SolType
from ...ethereum import Method
from ...symbolic.symbolic import constraints
from ...symbolic.symbolic import svm_utils
from ...symbolic.symbolic.world_state import WorldStateStatus
from ..random import policy_random
import logging
from copy import copy, deepcopy


class PolicySymPlus(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager):
        super().__init__(execution, contract_manager, account_manager)

        self.policy_random = policy_random.PolicyRandom(execution, contract_manager, account_manager)
        self.last_picked_pc_traces = []

        self.tx_count = 0
        self.idd_to_gstates = {}
        self.all_gstates = []

    def select_tx(self, obs):
        tx, idd = self.select_new_tx(obs)
        if tx is not None:
            return tx
        svm = obs.svm
        tx, idd = self.get_best_tx(obs, svm, self.all_gstates)
        if tx is not None:
            logging.info('jump to {}'.format(idd))
            self.jump_state(idd)
            svm.change_root(idd)
            return tx
        logging.info(f'no gain found globaly')
        return None

    def select_new_tx(self, obs):
        self.tx_count += 1
        svm = obs.svm
        gstates = []
        for address in svm.fuzz_addresses:
            gstates.extend(svm.sym_call_address(address, svm.root_wstate))
        for gstate in gstates:
            gstate.wstate_idd = obs.active_idd
        self.all_gstates.extend(gstates)
        self.idd_to_gstates[obs.active_idd] = gstates
        logging.info(f'found {len(gstates)} states')
        return self.get_best_tx(obs, svm, gstates)


    def get_best_tx(self, obs, svm, gstates):
        gain_to_gstates = defaultdict(list)
        for gstate in gstates:
            pc_set = set(gstate.pc_trace)
            gain = self.evaluate_pc_set_gain(obs.sym_stat, pc_set)
            gain_to_gstates[gain].append(gstate)
        for gain in sorted(gain_to_gstates.keys(), key=lambda k: -k):
            if gain == 0:
                logging.info('No feasible gain')
                return None, None
            # gstate = sorted(gain_to_gstates[gain], key=lambda g: len(g.pc_trace))[0]
            for gstate in sorted(gain_to_gstates[gain], key=lambda g: len(g.pc_trace)):
                if len(self.last_picked_pc_traces) and self.last_picked_pc_traces[-1] == gstate.pc_trace:
                    continue
                solver = self.get_state_solver(gstate)
                if solver is None:
                    continue
                model = solver.model()
                sender_value = model.eval(gstate.environment.sender).as_long()
                sender = svm.possible_caller_addresses.index(sender_value)
                amount = model.eval(gstate.environment.callvalue).as_long()
                method_name = gstate.wstate.trace.split('.')[1].split('(')[0]
                address = hex(gstate.environment.active_address)
                if address not in obs.contract_manager.address_to_contract:
                    raise Exception('unknown address')
                contract = obs.contract_manager.address_to_contract[address]
                timestamp = self._select_timestamp(obs)
                if method_name == 'fallback':
                    if Method.FALLBACK not in contract.abi.methods_by_name:
                        continue
                    method_name = Method.FALLBACK
                    self.add_pc_set_to_stat(obs.sym_stat, set(gstate.pc_trace))
                    logging.info(f'sending tx {method_name} {hex(sender_value)} {gain}')
                    return Tx(self, contract.name, address, method_name, bytes(), [], amount, sender, timestamp, True), gstate.wstate_idd
                method = contract.abi.methods_by_name[method_name]
                timestamp = model.eval(gstate.environment.timestamp).as_long()
                inputs = method.inputs
                arguments = []
                random_args = self.policy_random._select_arguments(contract, method, sender, obs)
                logging.info(f'sending tx {method.name} {hex(sender_value)} {gain}')
                for i, arg in enumerate(inputs):
                    using_random = False
                    t = arg.evm_type.t
                    arg_eval = None
                    calldata = svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                               gstate.wstate.gen,
                                                               index=4+i*32)
                    calldata_eval = model.eval(calldata)
                    if svm_utils.is_bv_concrete(calldata_eval):
                        arg_eval = calldata_eval.as_long()
                    else:
                        logging.debug(f'Using random variable for {method.name} {arg.name}')
                        using_random = True
                        arg_eval = random_args[i]
                    if not using_random:
                        if t == SolType.AddressTy:
                            caller_constraint = z3.Or([calldata == p for p in svm.possible_caller_addresses if p != sender_value])
                            solver.add(caller_constraint)
                            if solver.check() == z3.sat:
                                calldata_eval = solver.model().eval(calldata)
                                arg_eval = calldata_eval.as_long()
                            arg_eval = hex(arg_eval % (2**160))
                        elif t == SolType.FixedBytesTy:
                            arg_eval = arg_eval % (8 * arg.evm_type.size)
                            arg_bytes = arg_eval.to_bytes(arg.evm_type.size, 'big')
                            arg_eval = [int(b) for b in arg_bytes]
                        elif t == SolType.ArrayTy:
                            arg_eval = random_args[i]
                        elif t == SolType.BoolTy:
                            arg_eval = False if arg_eval == 0 else True
                        elif t == SolType.StringTy:
                            size = random.randint(int(math.log(arg_eval) / math.log(8)) + 1, 40)
                            arg_eval = arg_eval.to_bytes(size, 'big')
                            arg_eval = bytearray([c % 128 for c in arg_eval]).decode('ascii')
                    if not isinstance(arg_eval, type(random_args[i])):
                        arg_eval = random_args[i]
                    arguments.append(arg_eval)
                self.add_pc_set_to_stat(obs.sym_stat, set(gstate.pc_trace))
                tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
                return tx, gstate.wstate_idd
        return None, None


    @staticmethod
    def add_pc_set_to_stat(stat, pc_set):
        for contract_name, pc in pc_set:
            stat.covered_pcs_dict[contract_name].add(pc)


    @staticmethod
    def evaluate_pc_set_gain(stat, pc_set):
        covered_pcs_dict = deepcopy(stat.covered_pcs_dict)
        for contract_name, pc in pc_set:
            covered_pcs_dict[contract_name].add(pc)
        total_coverage = 0
        stat_total_coverage = 0
        for contract_name, coverages in covered_pcs_dict.items():
            total_coverage += len(coverages)
        for contract_name, coverages in stat.covered_pcs_dict.items():
            stat_total_coverage += len(coverages)
        return total_coverage - stat_total_coverage

    @staticmethod
    def get_state_solver(gstate):
        if gstate.wstate.status == WorldStateStatus.INFEASIBLE:
            return None
        solver = z3.Solver()
        solver.set('timeout',  3 * 60 * 1000)
        solver.add(gstate.wstate.constraints)
        res = solver.check()
        if res == z3.unknown: logging.info(f'{gstate.wstate.trace} gstate check timeout')
        gstate.wstate.status = WorldStateStatus.FEASIBLE if res == z3.sat else WorldStateStatus.INFEASIBLE
        return solver if res == z3.sat else None
