from .tx import Tx
from ..ethereum import *


class Log:

    def __init__(self, *args, **kwargs):
        self.pc = kwargs['pc']
        self.op = kwargs['op']
        self.gas = kwargs['gas']
        self.gas_cost = kwargs['gasCost']
        self.memory = kwargs['memory']
        self.memory_size = kwargs['memSize']
        self.stack = kwargs['stack']
        self.depth = kwargs['depth']
        self.op_name = kwargs['opName']
        self.error = kwargs['error']


class Logger:

    def __init__(self, *args, **kwargs):
        self.tx = Tx(**kwargs['tx']) if kwargs['tx'] is not None else None
        self.logs = [Log(**log) for log in kwargs['logs']] if kwargs['logs'] is not None else None
        self.bug_res = kwargs['bug_res']
        self.contract_receive_ether = kwargs['contract_receive_ether']


    def trace_log_memory(self, log_idx, low, high):
        memory_entries = set(range(low, high))
        value_from_call, value_from_block = False, False

        for i in range(log_idx, -1, -1):
            if len(memory_entries) == 0:
                break

            log = self.logs[i]
            stack = log.stack
            if log.op == MSTORE:
                offset = int(stack[-1], 16)
                cur_memory_entries = set(range(offset, offset+32))
                if len(memory_entries & cur_memory_entries) > 0:
                    memory_entries -= cur_memory_entries
                    v0, v1 = self.trace_log_stack(i-1, -2)
                    value_from_call |= v0
                    value_from_block |= v1
            elif log.op == MSTORE8:
                offset = int(stack[-1], 16)
                cur_memory_entries = set(range(offset, offset+8))
                if len(memory_entries & cur_memory_entries) > 0:
                    memory_entries -= cur_memory_entries
                    v0, v1 = self.trace_log_stack(i-1, -2)
                    value_from_call |= v0
                    value_from_block |= v1
            elif log.op == CALLDATACOPY:
                offset = int(stack[-1], 16)
                length = int(stack[-3], 16)
                cur_memory_entries = set(range(offset, offset+length))
                if len(memory_entries & cur_memory_entries) > 0:
                    memory_entries -= cur_memory_entries
                    if log.depth == 1:
                        value_from_call = True
                        value_from_block |= False
                    else:
                        for j in range(i, -1, -1):
                            if self.logs[j].depth < log.depth:
                                v0, v1 = self.trace_log_stack(j-1, -2)
                                value_from_call |= v0
                                value_from_block |= v1

                                if self.logs[j].op in (CALL, CALLCODE):
                                    args_offset = int(self.logs[j].stack[-4], 16)
                                    args_length = int(self.logs[j].stack[-5], 16)
                                    v0, v1 = self.trace_log_memory(j-1, args_offset, args_offset+args_length)
                                    value_from_call |= v0
                                    value_from_block |= v1
                                    break
                                elif self.logs[j].op in (DELEGATECALL, STATICCALL):
                                    args_offset = int(self.logs[j].stack[-3], 16)
                                    args_length = int(self.logs[j].stack[-4], 16)
                                    v0, v1 = self.trace_log_memory(j-1, args_offset, args_offset+args_length)
                                    value_from_call |= v0
                                    value_from_block |= v1
                                    break
                                else:
                                    assert False
                        else:
                            assert False
            elif log.op == CODECOPY:
                # TODO implement it
                pass
            elif log.op == EXTCODECOPY:
                v0, v1 = self.trace_log_stack(i-1, -1)
                value_from_call |= v0
                value_from_block |= v1
            elif log.op == RETURNDATACOPY:
                # TODO implement it
                pass
            elif log.op in (CALL, CALLCODE):
                ret_offset = int(stack[-6], 16)
                ret_length = int(stack[-7], 16)
                cur_memory_entries = set(range(ret_offset, ret_offset+ret_length))
                if len(memory_entries & cur_memory_entries) > 0:
                    memory_entries -= cur_memory_entries
                    v0, v1 = self.trace_log_stack(i-1, -1)
                    value_from_call |= v0
                    value_from_block |= v1
            elif log.op in (STATICCALL, DELEGATECALL):
                ret_offset = int(stack[-5], 16)
                ret_length = int(stack[-6], 16)
                cur_memory_entries = set(range(ret_offset, ret_offset+ret_length))
                if len(memory_entries & cur_memory_entries) > 0:
                    memory_entries -= cur_memory_entries
                    v0, v1 = self.trace_log_stack(i-1, -1)
                    value_from_call |= v0
                    value_from_block |= v1

        return value_from_call, value_from_block


    def trace_log_sotrage(self, log_idx, offset):
        for i in range(log_idx, -1, -1):
            log = self.logs[i]
            if log.op == SSTORE and int(log.stack[-1], 16) == offset:
                value_from_call, value_from_block = False, False 
                v0, v1 = self.trace_log_stack(i-1, -2)
                value_from_call |= v0
                value_from_block |= v1
                v0, v1 = self.trace_log_stack(i-1, -1)
                value_from_call |= v0
                value_from_block |= v1
                return value_from_call, value_from_block

        return False, False


    def trace_log_stack(self, log_idx, stack_pos):
        if log_idx == -1 or log_idx == len(self.logs) - 2:
            return False, False

        logs = self.logs
        log = logs[log_idx]
        op = log.op
        stack = log.stack

        if 0x60 <= op <= 0x7f:
            if stack_pos == -1:
                return False, False
            else:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op < 0x20:
            if op == STOP:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (ADD, MUL, SUB, DIV, SDIV, MOD, SMOD, EXP, SIGNEXTEND, \
                        LT, GT, SLT, SGT, EQ, AND, OR, XOR, BYTE, SHL, SHR, SAR):
                if stack_pos == -1:
                    value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
                    value_from_call1, value_from_block1 = self.trace_log_stack(log_idx-1, -2)
                    value_from_call = value_from_call0 | value_from_call1
                    value_from_block = value_from_block0 | value_from_block1
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (ADDMOD, MULMOD):
                if stack_pos == -1:
                    value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
                    value_from_call1, value_from_block1 = self.trace_log_stack(log_idx-1, -2)
                    value_from_call2, value_from_block2 = self.trace_log_stack(log_idx-1, -3)
                    value_from_call = value_from_call0 | value_from_call1 | value_from_call2
                    value_from_block = value_from_block0 | value_from_block1 | value_from_block2
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (ISZERO, NOT):
                if stack_pos == -1:
                    value_from_call, value_from_block = self.trace_log_stack(log_idx-1, -1)
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            else:
                assert False
        elif op < 0x40:
            if op == SHA3:
                if stack_pos == -1:
                    low, high = int(stack[-1], 16), int(stack[-1], 16) + int(stack[-2], 16)
                    v0, v1 = self.trace_log_memory(log_idx-1, low, high)
                    return v0, v1
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (ADDRESS, CODESIZE, RETURNDATASIZE):
                if stack_pos == -1:
                    return False, False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (ORIGIN, GASPRICE):
                if stack_pos == -1:
                    return True, False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == CALLER:
                if stack_pos == -1:
                    if log.depth == 1:
                        return True, False
                    else:
                        for i in range(log_idx, -1, -1):
                            if self.logs[i].depth < log.depth:
                                return self.trace_log_stack(i-1, -2)
                        else:
                            assert False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == CALLVALUE:
                if stack_pos == -1:
                    if log.depth == 1:
                        return True, False
                    else:
                        for i in range(log_idx, -1, -1):
                            if self.logs[i].depth < log.depth:
                                assert self.logs[i].op in (CALL, CALLCODE)
                                return self.trace_log_stack(i-1, -3)
                        else:
                            assert False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (CALLDATALOAD, CALLDATASIZE):
                if stack_pos == -1:
                    if log.depth == 1:
                        return True, False
                    else:
                        for i in range(log_idx, -1, -1):
                            if self.logs[i].depth < log.depth:
                                if self.logs[i].op in (CALL, CALLCODE):
                                    args_offset = int(self.logs[i].stack[-4], 16)
                                    args_length = int(self.logs[i].stack[-5], 16)
                                elif self.logs[i].op in (DELEGATECALL, STATICCALL):
                                    args_offset = int(self.logs[i].stack[-3], 16)
                                    args_length = int(self.logs[i].stack[-4], 16)
                                else:
                                    assert False
                                return self.trace_log_memory(i-1, args_offset, args_offset+args_length)
                        else:
                            assert False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == CALLDATACOPY:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (BALANCE, EXTCODESIZE):
                if stack_pos == -1:
                    value_from_call, value_from_block = self.trace_log_stack(log_idx-1, -1)
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (CALLDATACOPY, CODECOPY, RETURNDATACOPY):
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == EXTCODECOPY:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            else:
                assert False
        elif op < 0x50:
            if op == BLOCKHASH:
                if stack_pos == -1:
                    value_from_call, value_from_block = self.trace_log_stack(log_idx-1, -1)
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (COINBASE, TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT):
                if stack_pos == -1:
                    return False, True
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            else:
                assert False
        elif op < 0x60:
            if op == POP:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == MLOAD:
                if stack_pos == -1:
                    offset = int(stack[-1], 16)
                    return self.trace_log_memory(log_idx-1, offset, offset+32)
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op in (MSTORE, MSTORE8):
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == SLOAD:
                if stack_pos == -1:
                    value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
                    value_from_call1, value_from_block1 = self.trace_log_sotrage(log_idx-1, int(stack[-1], 16))
                    value_from_call = value_from_call0 | value_from_call1
                    value_from_block = value_from_block0 | value_from_block1
                    return value_from_call, value_from_block
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == SSTORE:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == JUMP:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == JUMPI:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == PC:
                if stack_pos == -1:
                    return False, False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == MSIZE:
                if stack_pos == -1:
                    return False, False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == GAS:
                if stack_pos == -1:
                    return False, False
                else:
                    return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            elif op == JUMPDEST:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
            else:
                assert False
        elif 0x80 <= op <= 0x8f: # DUP
            if stack_pos == -1:
                return self.trace_log_stack(log_idx-1, 0x7f-op)
            else:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif 0x90 <= op <= 0x9f: # SWAP
            if stack_pos == -1:
                return self.trace_log_stack(log_idx-1, 0x8e-op)
            elif stack_pos == 0x8e - op:
                return self.trace_log_stack(log_idx-1, -1)
            else:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif 0xa0 <= op <= 0xa4: # LOG
            return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op == CREATE:
            if stack_pos == -1:
                value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
                offset, length = int(log.stack[-2], 16), int(log.stack[-3], 16)
                value_from_call1, value_from_block1 = self.trace_log_memory(log_idx-1, offset, offset+length)
                value_from_call = value_from_call0 | value_from_call1
                value_from_block = value_from_block0 | value_from_block1
                return value_from_call, value_from_block
            else:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op in (CALL, CALLCODE):
            if stack_pos == -1:
                value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
                value_from_call1, value_from_block1 = self.trace_log_stack(log_idx-1, -2)
                value_from_call2, value_from_block2 = self.trace_log_stack(log_idx-1, -3)
                args_offset, args_length = int(stack[-4], 16), int(stack[-5], 16)
                value_from_call3, value_from_block3 = self.trace_log_memory(log_idx-1, args_offset, args_offset + args_length)
                value_from_call = (value_from_call0 | value_from_call1 | value_from_call2 | value_from_call3)
                value_from_block = (value_from_block0 | value_from_block1 | value_from_block2 | value_from_block3)

                return value_from_call, value_from_block
            else:
                return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op == RETURN:
            return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op in (DELEGATECALL, STATICCALL):
            value_from_call0, value_from_block0 = self.trace_log_stack(log_idx-1, -1)
            value_from_call1, value_from_block1 = self.trace_log_stack(log_idx-1, -2)
            args_offset, args_length = int(stack[-3], 16), int(stack[-4], 16)
            value_from_call2, value_from_block2 = self.trace_log_memory(log_idx-1, args_offset, args_offset + args_length)
            value_from_call = (value_from_call0 | value_from_call1 | value_from_call2)
            value_from_block = (value_from_block0 | value_from_block1 | value_from_block2)

            return value_from_call, value_from_block
        elif op == REVERT:
            return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op == INVALID:
            return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        elif op == SELFDESTRUCT:
            return self.trace_log_stack(log_idx-1, stack_pos+STACK_CHANGES[op])
        else:
            assert False, 'unsuppored opcode {}'.format(op)