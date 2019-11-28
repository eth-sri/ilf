import abc
import ethereum
from ..stat import Stat
import logging
from ..obs_base import ObsBase
from ...symbolic.symbolic import svm
from ...symbolic.solidity import asm
import json
import os
import glob


class ObsSymbolic(ObsBase):

    def __init__(self, contract_manager, account_manager, dataset_dump_path, backend_loggers):
        super().__init__(contract_manager, account_manager, dataset_dump_path)

        self.sym_stat = Stat(contract_manager, account_manager)

        self.contract_to_storage = {}
        self.tx_count = 0

        address_to_contract = self.contract_manager.address_to_contract
        fuzz_contract_names = self.contract_manager.fuzz_contract_names
        fuzz_addresses = [int(contract_manager.contract_dict[n].addresses[0][2:], 16) for n in fuzz_contract_names]

        self.hash_to_func_name = {}
        for contract in self.contract_manager.contract_dict.values():
            for method in contract.abi.methods:
                self.hash_to_func_name[method.idd] = method.name
        proj_path = self.contract_manager.proj_path
        build_dir = os.path.join(proj_path, 'build', 'contracts')
        build_json_files = glob.glob(os.path.join(build_dir,'*.json'))
        contract_to_build_data = {}
        for build_json_file in build_json_files:
            contract_name = os.path.splitext(os.path.basename(build_json_file))[0]
            with open(build_json_file) as f:
                build_data = json.loads(f.read())
            contract_to_build_data[contract_name] = build_data

        self.svm = svm.SVM(address_to_contract,
                           contract_to_build_data,
                           self.hash_to_func_name,
                           self.account_manager,
                           fuzz_addresses)

        contract_to_address = {v.name:k for k, v in address_to_contract.items()}
        for logger in backend_loggers:
            if logger.logs is not None:
                bytecode_bytes = self.get_create_bytecode(logger)
                if bytecode_bytes is None: continue
                found_swarmhashes = asm.find_swarmhashes(bytecode_bytes)
                create_contract = self.svm.swarm_hash_tuple_to_contract[tuple(found_swarmhashes)]
                if create_contract.name == 'Migrations': continue
                if create_contract.name not in contract_to_address: continue
                address = contract_to_address[create_contract.name]
                sstore_data, sha_data = self.get_logger_info(logger, address)
                self.svm.update_sha(sha_data)
                self.svm.update_storages(sstore_data)


    def init(self):
        super().init()


    def update(self, logger, is_init_explore):
        super().update(logger, is_init_explore)
        self.sym_stat.update(logger)
        self.tx_count += 1
        sstore_data, sha_data = self.get_logger_info(logger)
        self.svm.update_sha(sha_data)
        self.svm.update_storages(sstore_data)


    def get_create_bytecode(self, logger):
        log = logger.logs[-1]
        stack = log.stack
        if log.op_name != 'RETURN':
            return None
        offset = int(stack[-1][2:], 16)
        length = int(stack[-2][2:], 16)
        memory = '0' if log.memory == '0x' else log.memory[2:]
        memory_bytes = ethereum.utils.int_to_bytes(int(memory, 16))
        return_bytes = memory_bytes[offset:offset+length]
        return return_bytes


    def get_logger_info(self, logger, start_address=None):
        sha_data = []
        sstore_data = []
        assert logger.tx.call_address != '0x0' or start_address is not None
        start_address = logger.tx.call_address if start_address is None else start_address
        call_stack = [start_address]

        for i, log in enumerate(logger.logs):
            depth = log.depth
            stack = log.stack

            assert depth == len(call_stack)

            if log.op_name == 'SSTORE':
                index = int(stack[-1][2:], 16)
                value = int(stack[-2][2:], 16)
                sstore_data.append((call_stack[-1], index, value))
            elif log.op_name == 'SHA3':
                offset = int(stack[-1][2:], 16)
                length = int(stack[-2][2:], 16)
                memory = ethereum.utils.remove_0x_head(log.memory)
                memory_bytes = bytearray.fromhex(memory)
                arg_bytes = memory_bytes[offset:offset+length]
                arg = ethereum.utils.bytes_to_int(arg_bytes)
                next_log = logger.logs[i+1]
                value = int(next_log.stack[-1], 16)
                sha_data.append((arg, value, length))
            elif depth == 1 and (log.op_name == 'RETURN' or log.op_name == 'STOP'):
                return sstore_data, sha_data
            if i+1 < len(logger.logs) and logger.logs[i+1].depth == depth + 1:
                call_stack.append(call_stack[-1] if log.op_name == 'DELEGATECALL' else stack[-2])
            elif i+1 < len(logger.logs) and logger.logs[i+1].depth == depth - 1:
                call_stack.pop()
        return [], []
