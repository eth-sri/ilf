import os
import json
import numpy
import logging


from collections import OrderedDict
from ..solidity import ABI, Method
from .opcode import *
from .insn import Instruction
from .state import EVMState, is_top, all_not_top, Top, Value, StackChecker, TopCALLDATASIZE
from ..analysis import CFG
from .utils import TT256, TT256M1, TT255, SECP256K1P, to_signed, encode_int32, big_endian_to_int, sha3


LOG = logging.getLogger(__name__)


class ContractManager:

    def __init__(self, *args, **kwargs):
        self.proj_path = kwargs['proj_path']
        if 'contracts' in kwargs and kwargs['contracts'] is not None:
            self.contract_dict = dict([(name, Contract(**contract, manager=self)) for name, contract in kwargs['contracts'].items()])
        else:
            self.contract_dict = dict()
        self.fuzz_contract_names = list(sorted(self.contract_dict.keys()))

        self.address_to_contract = dict()
        for contract in self.contract_dict.values():
            for address in contract.addresses:
                self.address_to_contract[address] = contract


    def __getitem__(self, name):
        return self.contract_dict[name]


    def set_fuzz_contracts(self, contract_names):
        self.fuzz_contract_names = list(sorted(contract_names))


    def get_fuzz_contracts(self):
        return dict([(name, self.contract_dict[name]) for name in self.fuzz_contract_names])


    def is_payable(self, contract_name, method_name):
        return self.contract_dict[contract_name].is_payable(method_name)


    def dump(self, path):
        os.makedirs(path, exist_ok=True)

        for contract in self.contract_dict.values():
            contract.dump(path)


class Contract:

    def __init__(self, *args, **kwargs):
        self.manager = kwargs['manager']

        self.name = kwargs['name']
        self.addresses = kwargs['addresses']
        self.abi = ABI(contract=self, proj_path=self.manager.proj_path, payable=kwargs['payable'], **kwargs['abi'])

        self.can_receive_ether = False
        if len(self.abi.payable) > 0 and any(self.abi.payable.values()):
            self.can_receive_ether = True

        self.insns = [Instruction(**insn, contract=self) for insn in kwargs['insns']]
        self.insn_pc_to_idx = dict()
        for i, insn in enumerate(self.insns):
            self.insn_pc_to_idx[insn.pc] = i
            insn.idx = i

        self.propagate_state()

        self.cfg = CFG(self)

        self.can_send_ether = False
        for i, block in self.cfg.blocks.items():
            for j in range(i, i + block.length):
                insn = self.insns[j]
                if insn.op in (CREATE, CALL, CALLCODE, DELEGATECALL, STATICCALL, SELFDESTRUCT):
                    self.can_send_ether = True


    def to_json(self):
        j = OrderedDict()

        j['addresses'] = []
        j['addresses'] += self.addresses

        j['methods'] = OrderedDict()
        for method in self.abi.methods:
            j['methods'][method.name] = OrderedDict()
            j['methods'][method.name]['op_bow'] = method.bow
            for key, value in method.storage_args.items():
                j['methods'][method.name][key] = list(value)
        return j


    def __str__(self):
        return json.dumps(self.to_json())


    def __repr__(self):
        return '{} {}'.format(self.name, self.addresses)


    def dump(self, path):
        cfg_dot = self.cfg.to_graphviz()
        cfg_dot.render(directory=path)

        with open(os.path.join(path, '{}.evm'.format(self.name)), 'w') as disasm_file:
            disasm_file.write('bytecode for contract {}, project path {}'.format(self.name, self.manager.proj_path))
            for insn in self.insns:
                if insn.idx in self.cfg.blocks:
                    disasm_file.write('\n')
                disasm_file.write(str(insn) + '\n')


    def get_method_by_idd(self, idd):
        return self.abi.methods_by_idd[idd]


    def get_method_by_name(self, name):
        return self.abi.methods_by_name[name]


    def is_payable(self, method_name):
        return self.abi.payable[method_name]


    def propagate_state(self):
        state = EVMState()
        self._propagate_state(0, state, dict())


    def _propagate_state(self, insn_idx, state, visited_blocks):
        if insn_idx in visited_blocks and visited_blocks[insn_idx] >= 10:
            return

        state.block_trace = [insn_idx] + state.block_trace

        if insn_idx not in visited_blocks:
            visited_blocks[insn_idx] = 0
        visited_blocks[insn_idx] += 1

        for insn_idx in range(insn_idx, len(self.insns)):
            insn = self.insns[insn_idx]
            op = insn.op
            
            insn.add_state(state.copy())

            with StackChecker(insn, state):
                if 0x60 <= op <= 0x7f:
                    state.push_stack(Value(insn.arg))
                elif op < 0x10:
                    if op == STOP:
                        break
                    elif op == ADD:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = (s0 + s1) & TT256M1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == MUL:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = (s0 * s1) & TT256M1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SUB:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = (s0 - s1) & TT256M1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == DIV:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = 0 if s1 == 0 else (s0 // s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SDIV:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            s0, s1 = to_signed(s0), to_signed(s1)
                            res = 0 if s1 == 0 else ((abs(s0) // abs(s1) * (-1 if s0 * s1 < 0 else 1)) & TT256M1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == MOD:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = 0 if s1 == 0 else (s0 % s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SMOD:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            s0, s1 = to_signed(s0), to_signed(s1)
                            res = 0 if s1 == 0 else ((abs(s0) % abs(s1) * (-1 if s0 < 0 else 1)) & TT256M1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == ADDMOD:
                        s0, s1, s2 = state.pop_stack(3)
                        if all_not_top((s0, s1, s2)):
                            res = ((s0 + s1) % s2) if s2 else 0
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == MULMOD:
                        s0, s1, s2 = state.pop_stack(3)
                        if all_not_top((s0, s1, s2)):
                            res = ((s0 * s1) % s2) if s2 else 0
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == EXP:
                        base, exponent = state.pop_stack(2)
                        if all_not_top((base, exponent)):
                            res = pow(base, exponent, TT256)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SIGNEXTEND:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            if s0 <= 31:
                                testbit = s0 * 8 + 7
                                if s1 & (1 << testbit):
                                    res = s1 | (TT256 - (1 << testbit))
                                else:
                                    res = s1 & ((1 << testbit) - 1)
                            else:
                                res = s1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    else:
                        assert False
                elif op < 0x20:
                    if op == LT:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 < s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == GT:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 > s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SLT:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            s0, s1 = to_signed(s0), to_signed(s1)
                            res = int(s0 < s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SGT:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            s0, s1 = to_signed(s0), to_signed(s1)
                            res = int(s0 > s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == EQ:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 == s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == ISZERO:
                        s0 = state.pop_stack()
                        if not is_top(s0):
                            res = int(s0 == 0)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == AND:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 & s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == OR:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 | s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == XOR:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            res = int(s0 ^ s1)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == NOT:
                        s0 = state.pop_stack()
                        if not is_top(s0):
                            res = TT256M1 - s0
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == BYTE:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            if s0 >= 32:
                                res = 0
                            else:
                                res = (s1 // 256 ** (31 - s0)) % 256
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SHL:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            if s0 >= 256:
                                res = 0
                            else:
                                res = numpy.left_shift(s1, s0) & TT256M1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SHR:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            if s0 >= 256:
                                res = 0
                            else:
                                res = numpy.right_shift(s1, s0) & TT256M1
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == SAR:
                        s0, s1 = state.pop_stack(2)

                        if all_not_top((s0, s1)):
                            if s0 >= 256:
                                if s1 > 0:
                                    res = 0
                                else:
                                    res = -1
                            else:
                                res = (s1 >> s0) & TT256M1
                        else:
                            state.push_stack(Top())
                    else:
                        assert False
                elif op < 0x40:
                    if op == SHA3:
                        s0, s1 = state.pop_stack(2)
                        if all_not_top((s0, s1)):
                            if state.mem_valid(s0, s1):
                                res = big_endian_to_int(sha3(state.mem_load(s0, s1)))
                                state.push_stack(Value(res))
                            else:
                                state.push_stack(Top())
                        else:
                            state.push_stack(Top())
                    elif op in (ADDRESS, ORIGIN, CALLER, CALLVALUE, CODESIZE, GASPRICE, RETURNDATASIZE):
                        state.push_stack(Top())
                    elif op in (BALANCE, EXTCODESIZE):
                        state.pop_stack()
                        state.push_stack(Top())
                    elif op == CALLDATASIZE:
                        state.push_stack(TopCALLDATASIZE())
                    elif op == CALLDATALOAD:
                        state.pop_stack()
                        state.push_stack(Top())
                    elif op in (CALLDATACOPY, CODECOPY, RETURNDATACOPY):
                        dest_offset, _, length = state.pop_stack(3)
                        if all_not_top((dest_offset, length)) and dest_offset + length < 0x1000:
                            state.mem_extend(dest_offset + length)
                            state.mem_reset(dest_offset, length)
                        else:
                            state.mem_reset_all()
                    elif op == EXTCODECOPY:
                        _, dest_offset, _, length = state.pop_stack(4)
                        if all_not_top((dest_offset, length)) and dest_offset + length < 0x1000:
                            state.mem_extend(dest_offset + length)
                            state.mem_reset(dest_offset, length)
                        else:
                            state.mem_reset_all()
                    else:
                        assert False
                elif op < 0x50:
                    if op == BLOCKHASH:
                        state.pop_stack()
                        state.push_stack(Top())
                    elif op in (COINBASE, TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT):
                        state.push_stack(Top())
                    else:
                        assert False
                elif op < 0x60:
                    if op == POP:
                        state.pop_stack()
                    elif op == MLOAD:
                        offset = state.pop_stack()
                        if not is_top(offset) and state.mem_valid(offset, 0x20):
                            res = state.mem_load(offset, 0x20)
                            state.push_stack(Value(res))
                        else:
                            state.push_stack(Top())
                    elif op == MSTORE:
                        offset, value = state.pop_stack(2)
                        if not is_top(offset) and offset < 0x1000:
                            if not is_top(value):
                                state.mem_store(offset, encode_int32(value))
                            else:
                                state.mem_store_top(offset, 0x20)
                        else:
                            state.mem_reset_all()
                    elif op == MSTORE8:
                        offset, value = state.pop_stack(2)
                        if not is_top(offset) and offset < 0x1000:
                            if not is_top(value):
                                state.mem_store(offset, encode_int32(value)[:1])
                            else:
                                state.mem_store_top(offset, 8)
                        else:
                            state.mem_reset_all()
                    elif op == SLOAD:
                        key = state.pop_stack()
                        if not is_top(key):
                            if key in state.storage:
                                res = state.storage[key]
                                if not is_top(res):
                                    state.push_stack(Value(res.value))
                                else:
                                    state.push_stack(Top())
                            else:
                                state.push_stack(Top())
                        else:
                            state.push_stack(Top())
                    elif op == SSTORE:
                        key, value = state.pop_stack(2)
                        if not is_top(key):
                            if not is_top(value):
                                state.storage[key] = Value(value)
                            else:
                                state.storage[key] = Top()
                        else:
                            state.storage.clear()
                    elif op == JUMP:
                        pc = state.pop_stack()
                        if not is_top(pc):
                            self._propagate_state(self.insn_pc_to_idx[pc], state.copy(), visited_blocks)
                        else:
                            LOG.debug('for contract {} of project path {}, insn {} target has argument as TOP'.format(self.name, self.manager.proj_path, insn))
                        break
                    elif op == JUMPI:
                        pc, _ = state.pop_stack(2)
                        if not is_top(pc):
                            self._propagate_state(self.insn_pc_to_idx[pc], state.copy(), visited_blocks)
                        else:
                            LOG.debug('for contract {} of project path {}, insn {} target has argument as TOP'.format(self.name, self.manager.proj_path, insn))

                        self._propagate_state(insn_idx + 1, state.copy(), visited_blocks)
                        break
                    elif op == PC:
                        state.push_stack(Value(insn.pc))
                    elif op == MSIZE:
                        state.push_stack(state.mem_size())
                    elif op == GAS:
                        state.push_stack(Top())
                    elif op == JUMPDEST:
                        if insn_idx not in visited_blocks:
                            visited_blocks[insn_idx] = 0
                        visited_blocks[insn_idx] += 1
                    else:
                        assert False
                elif 0x80 <= op <= 0x8f: # DUP
                    stack = state.stack
                    state.push_stack(stack[0x7f - op].copy())
                elif 0x90 <= op <= 0x9f: # SWAP
                    stack = state.stack
                    tmp = stack[0x8e - op]
                    stack[0x8e - op] = stack[-1].copy()
                    stack[-1] = tmp.copy()
                elif 0xa0 <= op <= 0xa4: # LOG
                    if op == LOG0:
                        state.pop_stack(2)
                    elif op == LOG1:
                        state.pop_stack(3)
                    elif op == LOG2:
                        state.pop_stack(4)
                    elif op == LOG3:
                        state.pop_stack(5)
                    elif op == LOG4:
                        state.pop_stack(6)
                    else:
                        assert False
                elif op == CREATE:
                    state.pop_stack(3)
                    state.push_stack(Top())
                elif op in (CALL, CALLCODE):
                    _, _, _, _, _, ret_offset, ret_length = state.pop_stack(7)
                    if all_not_top((ret_offset, ret_length)):
                        state.mem_reset(ret_offset, ret_length)
                    else:
                        state.mem_reset_all()
                    state.push_stack(Top())
                elif op == RETURN:
                    state.pop_stack(2)
                    break
                elif op in (DELEGATECALL, STATICCALL):
                    _, _, _, _, ret_offset, ret_length = state.pop_stack(6)
                    if all_not_top((ret_offset, ret_length)):
                        state.mem_reset(ret_offset, ret_length)
                    else:
                        state.mem_reset_all()
                    state.push_stack(Top())
                elif op == REVERT:
                    state.pop_stack(2)
                    break
                elif op == INVALID:
                    break
                elif op == SELFDESTRUCT:
                    state.pop_stack()
                    break
                else:
                    assert False, 'unsuppored opcode {}'.format(op)