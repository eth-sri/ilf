from ilf.symbolic import utils
from ilf.symbolic.solidity import asm
from ilf.symbolic.solidity.soliditycontract import SolidityContract
from ilf.symbolic.utils import BColors, ADDRESS_ARG_TAG
from ilf.symbolic.symbolic import svm_utils
from ilf.symbolic.symbolic.account import Account, AccountType
from ilf.symbolic.symbolic.world_state import WorldState, WorldStateStatus
from ilf.symbolic.symbolic.environment import CalldataType, Environment
from ilf.symbolic.symbolic.global_state import GlobalState
from ilf.symbolic.symbolic.svm_utils import TT256, TT256M1, make_trace, VECTOR_LEN
from ilf.symbolic.symbolic.svm_utils import extract_trace_to_independent_traces
from ilf.symbolic.symbolic.svm_utils import SIMPLE_STORAGE_SYMBOLIC_TAG, MAPPING_STORAGE_PREFIX, EMPTY_ARRAY
from ilf.symbolic.exceptions import DeploymentError, SVMRuntimeError
from ilf.symbolic.symbolic import constraints
from ilf.symbolic.symbolic import storage
from ilf.symbolic.symbolic.execution import Executor
from ilf.symbolic.solidity import solidity_utils
from collections import defaultdict
import pprint
import os
import json
from enum import Enum
import itertools
from web3 import Web3
from copy import copy, deepcopy
import ethereum
import logging
import multiprocessing
import pickle
import time
import random
import sys
from z3 import BitVecVal, BitVec, BitVecSort, BitVecNumRef, BoolVal
from z3 import Concat, Extract, UDiv, URem, ULT, UGT, UGE, sat, unsat, Solver
from z3 import If, simplify, Not, BoolRef, is_false, is_true, Select, K, Store, Array
import z3
import z3.z3util

from ilf.ethereum import Method
from ilf.ethereum import SolType


DEPLOYER_CONTRACT_NAME = 'Deployer'
DEPLOYER_ADDRESS = 0xC0FFEE
AUTO_PREDICATES = False

SYMBOLIC_CALL_RETURNDATA_SIZE_BYTES = 32
CALLER_POOL_SIZE = 5
# SYMBOLIC_SHA_PADDING should be < SYMBOLIC_FIELD_PADDING
SYMBOLIC_SHA_PADDING = 1024
SYMBOLIC_FIELD_PADDING = 2048
SE_DEPTH = 2
SYM_EXEC_LIMIT = 20 * 60
CONCRETE_START_TIMESTAMP = False

EACH_OP_ABSTRACTION = False
JUMP_ABSTRACTION = False


PRINT_STACK = False
# TODO: add constraint for name/hash of function executed; consider 'hash' values for fallback and init

class CallerType(Enum):
    CALLER_CONCRETE = 1
    CALLER_SYMBOLIC = 2

SHA_EMPTY_ARGS_VALUE = 2**255 - 0x12345

class SVM:
    def __init__(self,
                 address_to_contract,
                 contract_to_build_data,
                 hash_to_func_name,
                 account_manager,
                 fuzz_addresses,
                 abs_mul=False,
                 abs_div=False,
                 max_jump_depth=3):
        self.account_manager = account_manager
        caller_accounts = account_manager.accounts

        self.contract_to_build_data = contract_to_build_data
        self.address_to_contract = address_to_contract
        self.max_jump_depth = max_jump_depth
        self.fuzz_addresses = fuzz_addresses

        self.hash_to_func_name = hash_to_func_name
        self.debug = False

        # Core fields
        self.idd_to_wstate = {}
        self.root_wstate = WorldState()
        self.idd_to_wstate[0] = self.root_wstate
        self.swarm_hash_to_contract = {}
        self.gen_to_wstates = defaultdict(list)
        self.sym_bv_generator = constraints.SymbolicBitvecGenerator()
        self.contract_name_to_contract = {}
        self.abs_mul = abs_mul
        self.abs_div = abs_div


        self.explored_hashes = []
        self.storage_indexes = []
        # DEBUG
        self.reduced_traces = []

        # yaml delpolment related

        self.executor = Executor(self)

        self.sum_references = []
        self.seen_predicate_signatures = set()

        self.storage_root_loaded = False
        self.seen_hashes = []


        self.swarm_hash_tuple_to_contract = {}
        self.deploy_contracts()
        self.log_sha_to_sym_sha = {}
        self.possible_caller_addresses = [int(a.address[2:], 16) for a in caller_accounts]


        logging.info('SVM initialized')

    def copy_root(self, idd):
        root_copy = copy(self.root_wstate)
        self.idd_to_wstate[idd] = root_copy


    def change_root(self, idd):
        self.root_wstate = self.idd_to_wstate[idd]

    def update_wstate(self, idd, sha_data, sstore_data):
        self.update_sha(sha_data)
        self.update_storages(sstore_data)

    def update_storages(self, sstore_data):
        for address, index, value in sstore_data:
            address = int(address[2:], 16)
            assert address in self.root_wstate.address_to_account
            if index > 100:
                if index in self.log_sha_to_sym_sha:
                    store_index = self.log_sha_to_sym_sha[index]
                else:
                    log_shas = list(self.log_sha_to_sym_sha.keys())
                    diffs = [abs(l - index) for l in log_shas]
                    min_index = diffs.index(min(diffs))
                    diff = diffs[min_index]
                    if diff < 10:
                        relative_index = log_shas[min_index]
                        store_index = z3.simplify(self.log_sha_to_sym_sha[relative_index] + z3.BitVecVal(diff, 256))
                        self.log_sha_to_sym_sha[index] = store_index
                    else:
                        store_index = z3.BitVecVal(index, 256)
            else:
                store_index = z3.BitVecVal(index, 256)
            store_value = self.log_sha_to_sym_sha.get(value, z3.BitVecVal(value, 256))
            account = self.root_wstate.address_to_account[address]
            account.storage.store(store_index, store_value)

    def update_sha(self, sha_data):
        for arg, log_value, length_bytes in sha_data:
            if log_value in self.log_sha_to_sym_sha:
                continue
            data = z3.BitVecVal(arg, length_bytes * 8)
            if data.size() == 512:
                data_words = svm_utils.split_bv_by_words(data)
                data_words = [d.as_long() for d in data_words]
                data_words = [self.log_sha_to_sym_sha.get(d, z3.BitVecVal(d, 256)) for d in data_words]
                data = z3.simplify(z3.Concat(data_words))
            sha_constraints, hash_vector = svm_utils.symbolic_keccak(self, data)
            self.log_sha_to_sym_sha[log_value] = hash_vector
            self.root_wstate.constraints.extend(sha_constraints)
        solver = z3.Solver()
        solver.add(self.root_wstate.constraints)
        assert solver.check() == z3.sat


    def trim_unrechable_states(self):
        # (parent, trace, child) tuples
        pending_parent_trace_child_tuples = [(None, None, self.root_wstate)]
        deleted_counter = 0
        s = Solver()
        while(len(pending_parent_trace_child_tuples)):
            s.push()
            parent_wstate, trace, curr_wstate = pending_parent_trace_child_tuples.pop()
            if curr_wstate.status != WorldStateStatus.REACHABLE:
                s.add(curr_wstate.constraints)
                res = s.check()
                if res == sat:
                    curr_wstate.status = WorldStateStatus.REACHABLE
                elif res == unsat:
                    curr_wstate.status = WorldStateStatus.UNREACHABLE
                elif res == z3.unknown:
                    print(curr_wstate.get_full_trace())
                    raise Exception("pdb")
            if curr_wstate.status == WorldStateStatus.REACHABLE:
                if curr_wstate != self.root_wstate:
                    parent_wstate.trace_to_children[trace].append(curr_wstate)
                for child_trace, children in curr_wstate.trace_to_children.items():
                    for child_wstate in children:
                        pending_parent_trace_child_tuples.append((curr_wstate, child_trace, child_wstate))
                curr_wstate.trace_to_children.clear()
            else:
                curr_wstate.status = WorldStateStatus.DELETED
                self.gen_to_wstates[curr_wstate.gen].remove(curr_wstate)
                deleted_counter += 1
            s.pop()
        logging.info('%d WorldStates are deleted', deleted_counter)

        logging.info('SVM initialized')

    def check_subsumption(self, wstate):
        """
        @param wstate
        @returns: True if the wstate is subsumed
        """
        for p, not_p in wstate.predicate_tuples:
            if p and not_p:
                return True
        p_tuple = tuple(itertools.chain(wstate.predicate_tuples))
        if p_tuple is self.seen_predicate_signatures:
            return True
        subsumed = False
        for other_p_tuple in self.seen_predicate_signatures:
            if self.check_tuples_subsumption(other_p_tuple, p_tuple):
                subsumed = True
                break
        self.seen_predicate_signatures.add(p_tuple)
        return subsumed

    def check_tuples_subsumption(self, p_tuple_a, p_tuple_b):
        res = True
        for p_a, p_b in zip(itertools.chain(*p_tuple_a), itertools.chain(*p_tuple_b)):
            res = res and (not(p_a) or p_b)
        assert(type(res) == bool)
        return res

    def deploy_contracts(self):
        # Gathering existing libraries
        self.contract_name_to_contract, lib_name_to_address = svm_utils.generate_contract_objects(self.contract_to_build_data, self.hash_to_func_name)
        self.swarm_hash_to_contract = svm_utils.resolve_swarmhashes(self.contract_name_to_contract)
        for contract in self.contract_name_to_contract.values():
            swarm_hashes = tuple(contract.creation_disassembly.swarm_hashes)
            self.swarm_hash_tuple_to_contract[swarm_hashes] = contract
        for address, contract in self.address_to_contract.items():
            int_address = int(address[2:], 16)
            account = Account(int_address, self.contract_name_to_contract[contract.name])
            self.root_wstate.address_to_account[int_address] = account
        logging.info(f'Deployed contracts: {pprint.pformat(self.root_wstate.address_to_account)}')
        self.gen_to_wstates[self.root_wstate.gen].append(self.root_wstate)
        timestamp_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.TIMESTAMP,
                                                             self.root_wstate.gen)
        timestamp_constraint = ULT(1438214400, timestamp_vec)
        self.root_wstate.constraints.append(timestamp_constraint)

    def sym_call_address(self, address, wstate, method=None, arguments=None, amount=None, timestamp=None):
        # Initialize the execution environment
        account = wstate.address_to_account[address]
        if account.typ == AccountType.LIBRARY:
            return []
        child_wstate = wstate.child()
        logging.debug(BColors.GREEN + BColors.REVERSED + 'Starting SVM execution' + BColors.ENDC
                      + ' with address: ' + hex(address)
                      + ' contract: ' + account.contract.name)

        caller_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLER,
                                                          child_wstate.gen)

        caller_constraint = z3.Or([caller_vec == p for p in self.possible_caller_addresses])
        child_wstate.constraints.append(caller_constraint)

        calldata = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA_ARRAY,
                                                        child_wstate.gen,
                                                        acc=account.id)
        if method is not None and method.name != Method.FALLBACK:
            calldata_0 = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                              child_wstate.gen,
                                                              index=0)
            calldata_4bytes, _  = svm_utils.split_bv(calldata_0, calldata_0.size() - 32)
            calldatasize = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATASIZE,
                                                                child_wstate.gen,
                                                                acc=account.id)
            entry_account = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.ENTRY_ACCOUNT,
                                                                 child_wstate.gen)
            method_constraint = z3.And(calldata_4bytes == method.idd, z3.ULE(4, calldatasize), entry_account == address)
            child_wstate.constraints.append(method_constraint)
            if arguments is not None and len(arguments):
                encoded_args = utils.abi_encode(method.inputs, arguments)
                for i, arg in enumerate(encoded_args):
                    if arg is None:
                        continue
                    calldata_i = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                                      child_wstate.gen,
                                                                      index=4+32*i)
                    child_wstate.constraints.append(calldata_i == z3.BitVecVal(arg, 256))
        gasprice_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.GASPRICE,
                                                            child_wstate.gen)
        callvalue_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLVALUE,
                                                             child_wstate.gen)
        origin_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.ORIGIN,
                                                          child_wstate.gen)
        timestamp_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.TIMESTAMP,
                                                             child_wstate.gen)
        parent_timestamp_vec = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.TIMESTAMP,
                                                                    wstate.gen)
        callvalue_constraint = ULT(callvalue_vec, svm_utils.ETHER_LIMIT)
        timestamp_constraint = ULT(parent_timestamp_vec, timestamp_vec)

        entry_account = self.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.ENTRY_ACCOUNT,
                                                             child_wstate.gen)
        child_wstate.constraints.append(callvalue_constraint)
        child_wstate.constraints.append(timestamp_constraint)
        child_wstate.constraints.append(entry_account == account.address)
        environment = Environment(active_address=address,
                                  sender=caller_vec,
                                  calldata=calldata,
                                  gasprice=gasprice_vec,
                                  callvalue=callvalue_vec,
                                  origin=origin_vec,
                                  calldata_type=CalldataType.UNDEFINED,
                                  disassembly=account.contract.disassembly,
                                  runtime_bytecode_bytes=list(account.contract.disassembly.bytecode),
                                  timestamp=timestamp_vec)

        gstate = GlobalState(child_wstate, environment)
        return self.executor.execute_gstate(gstate)
