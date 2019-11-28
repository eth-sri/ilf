import numpy
from graphviz import Digraph

from ..evm.opcode import *
from ..evm.state import is_top, TopCALLDATASIZE
from ..solidity import Method
from ..evm.utils import TT256, TT256M1, TT255, SECP256K1P, to_signed, encode_int32, big_endian_to_int, sha3


BLOCK_VISIT_LIMIT = 1


class CFG:
    
    def __init__(self, contract):
        self.contract = contract

        self.blocks = dict()
        self.init_blocks()

        self.insn_idx_to_block = dict()
        for block in self.blocks.values():
            for i in range(block.start_idx, block.start_idx + block.length):
                self.insn_idx_to_block[i] = block

        self.method_to_entry = dict()
        self.entry_to_method = dict()
        self.init_method_entry()
        for method_name, entry_block_idx in self.method_to_entry.items():
            method = self.contract.get_method_by_name(method_name)
            entry_block = self.blocks[entry_block_idx]
            entry_block.add_block_to_method(method)
            method.bow = select_interesting_ops(method.bow)

        self.init_methods_feature()


    def init_methods_feature(self):
        for method_name, entry_block_idx in self.method_to_entry.items():
            method = self.contract.get_method_by_name(method_name)
            entry_block = self.blocks[entry_block_idx]
            result = dict()
            entry_block.storage_args(method, result, dict())
            method.storage_args = result
            # print(method_name)
            # print(result)


    def init_method_entry(self):
        insns = self.contract.insns

        for insn_idx, insn in enumerate(insns):
            if insn.op not in (EQ, LT) or len(insn.states) > 1 or insn_idx not in self.insn_idx_to_block:
                continue

            block = self.insn_idx_to_block[insn_idx]
            last_insn = block.last_insn
            if last_insn.op != JUMPI \
                    or len(last_insn.states) != 1 \
                    or len(list(last_insn.states)[0].stack) == 0 \
                    or is_top(list(last_insn.states)[0].stack[-1]):
                continue

            state = list(insn.states)[0]
            s0, s1 = state.stack[-1], state.stack[-2]
            if is_top(s0) and is_top(s1):
                continue

            if insn.op == EQ:
                if is_top(s0) and s1.value in self.contract.abi.methods_by_idd:
                    method = self.contract.get_method_by_idd(s1.value)
                    entry_pc = list(last_insn.states)[0].stack[-1].value
                    entry_idx = self.contract.insn_pc_to_idx[entry_pc]
                    self.method_to_entry[method.name] = entry_idx
                    self.entry_to_method[entry_idx] = method
                    method.entry_block = entry_idx

                if is_top(s1) and s0.value in self.contract.abi.methods_by_idd:
                    method = self.contract.get_method_by_idd(s0.value)
                    entry_pc = list(last_insn.states)[0].stack[-1].value
                    entry_idx = self.contract.insn_pc_to_idx[entry_pc]
                    self.method_to_entry[method.name] = entry_idx
                    self.entry_to_method[entry_idx] = method
                    method.entry_block = entry_idx

            if insn.op == LT and Method.FALLBACK in self.contract.abi.methods_by_name:
                if s0.__class__ == TopCALLDATASIZE and s1.value == 4:
                    method = self.contract.get_method_by_name(Method.FALLBACK)
                    entry_pc = list(last_insn.states)[0].stack[-1].value
                    entry_idx = self.contract.insn_pc_to_idx[entry_pc]
                    self.method_to_entry[Method.FALLBACK] = entry_idx
                    self.entry_to_method[entry_idx] = method
                    method.entry_block = entry_idx


    def init_blocks(self):
        self.add_block(0)

        insns = self.contract.insns
        block_idx = 0
        for insn_idx, insn in enumerate(insns):
            # print(insn)
            # for state in insn.states:
            #     print(state)
            if insn.op in (STOP, RETURN, SELFDESTRUCT, REVERT, INVALID):
                block = self.add_block(block_idx)
                block.length = insn_idx - block_idx + 1

                block_idx = insn_idx + 1
            elif insn.op == JUMP:
                block = self.add_block(block_idx)
                block.length = insn_idx - block_idx + 1

                # assert len(insn.states) > 0, 'insn [{}] has zero number of states'.format(str(insn))
                jump_dests = set()
                for state in insn.states:
                    jump_dest = state.stack[-1]
                    if not is_top(jump_dest):
                        jump_dests.add(jump_dest.value)

                for jump_dest in jump_dests:
                    branch_idx = self.contract.insn_pc_to_idx[jump_dest]
                    self.add_edge(block_idx, branch_idx, True, len(jump_dests) > 1)

                block_idx = insn_idx + 1
            elif insn.op == JUMPI:
                block = self.add_block(block_idx)
                block.length = insn_idx - block_idx + 1

                default_idx = insn_idx + 1
                self.add_edge(block_idx, default_idx, False, False)

                # assert len(insn.states) > 0, 'insn [{}] has zero number of states'.format(str(insn))
                jump_dests = set()
                for state in insn.states:
                    jump_dest = state.stack[-1]
                    if not is_top(jump_dest):
                        jump_dests.add(jump_dest.value)

                for jump_dest in jump_dests:
                    branch_idx = self.contract.insn_pc_to_idx[jump_dest]
                    self.add_edge(block_idx, branch_idx, True, len(jump_dests) > 1)

                block_idx = insn_idx + 1
            elif insn.op == JUMPDEST:
                if insn_idx > 0 \
                        and insns[insn_idx - 1].op not in \
                        (STOP, JUMP, JUMPI, RETURN, REVERT, INVALID, SELFDESTRUCT):
                    block = self.add_block(block_idx)
                    block.length = insn_idx - block_idx

                    default_idx = insn_idx
                    self.add_edge(block_idx, default_idx, False, False)

                    block_idx = insn_idx

        for block in self.blocks.values():
            if block.length is None:
                for i in range(block.start_idx, len(insns)):
                    if i in self.blocks:
                        break

                block.length = i - block.start_idx

            block.last_insn = insns[block.start_idx + block.length - 1]


    def add_edge(self, src_idx, tgt_idx, branch_or_default, is_dep):
        src_block = self.add_block(src_idx)
        tgt_block = self.add_block(tgt_idx)

        if is_dep:
            tgt_block.dep_parents.add(src_idx)

            if branch_or_default:
                src_block.dep_branches.add(tgt_idx)
            else:
                src_block.dep_defaults.add(tgt_idx)
        else:
            tgt_block.parents.add(src_idx)

            if branch_or_default:
                src_block.branch = tgt_idx
            else:
                src_block.default = tgt_idx


    def add_block(self, start_idx):
        if start_idx not in self.blocks:
            block = CFGBlock(start_idx, self)
            self.blocks[start_idx] = block

        return self.blocks[start_idx]


    def to_graphviz(self):
        insns = self.contract.insns

        graph = Digraph(name=self.contract.name, format='svg')
        for start_idx in sorted(self.blocks.keys()):
            block = self.blocks[start_idx]
            last_insn = block.last_insn

            if last_insn.op == JUMP:
                if start_idx in self.entry_to_method:
                    label = '{} {} {} {}'.format(self.entry_to_method[start_idx].name, start_idx, format(insns[start_idx].pc, '02x'), block.length)
                else:
                    label = '{} {} {}'.format(start_idx, format(insns[start_idx].pc, '02x'), block.length)

                node_attrs = {}
                if not block.covered:
                    node_attrs['fillcolor'] = 'red'
                    node_attrs['style'] = 'filled'

                if start_idx in self.entry_to_method:
                    node_attrs['fillcolor'] = 'yellow'
                    node_attrs['style'] = 'filled'

                graph.node(name=str(start_idx), label=label, **node_attrs)

                if block.branch is not None:
                    graph.edge(str(start_idx), str(block.branch), color='black')

                for dep_branch in block.dep_branches:
                    graph.edge(str(start_idx), str(dep_branch), color='black', style='dashed')
            elif last_insn.op == JUMPI:
                if start_idx in self.entry_to_method:
                    label = '{} {} {} {}'.format(self.entry_to_method[start_idx].name, start_idx, format(insns[start_idx].pc, '02x'), block.length)
                else:
                    label = '{} {} {}'.format(start_idx, format(insns[start_idx].pc, '02x'), block.length)

                node_attrs = {}
                if not block.covered:
                    node_attrs['fillcolor'] = 'red'
                    node_attrs['style'] = 'filled'

                if start_idx in self.entry_to_method:
                    node_attrs['fillcolor'] = 'yellow'
                    node_attrs['style'] = 'filled'

                graph.node(name=str(start_idx), label=label, **node_attrs)

                if block.branch is not None:
                    graph.edge(str(start_idx), str(block.branch), color='green')
                if block.default is not None:
                    graph.edge(str(start_idx), str(block.default), color='red')

                for dep_branch in block.dep_branches:
                    graph.edge(str(start_idx), str(dep_branch), color='green', style='dashed')
                for dep_default in block.dep_defaults:
                    graph.edge(str(start_idx), str(dep_default), color='red', style='dashed')
            elif last_insn.op in (STOP, RETURN, SELFDESTRUCT, REVERT, INVALID):
                if start_idx in self.entry_to_method:
                    label = '{} {} {} {}'.format(self.entry_to_method[start_idx].name, start_idx, format(insns[start_idx].pc, '02x'), block.length)
                else:
                    label = '{} {} {}'.format(start_idx, format(insns[start_idx].pc, '02x'), block.length)

                node_attrs = {}
                node_attrs['style'] = 'filled'
                if not block.covered:
                    node_attrs['fillcolor'] = 'purple'
                else:
                    node_attrs['fillcolor'] = 'blue'

                if start_idx in self.entry_to_method:
                    node_attrs['fillcolor'] = 'yellow'

                graph.node(name=str(start_idx), label=label, **node_attrs)
            else:
                if start_idx in self.entry_to_method:
                    label = '{} {} {} {}'.format(self.entry_to_method[start_idx].name, start_idx, format(insns[start_idx].pc, '02x'), block.length)
                else:
                    label = '{} {} {}'.format(start_idx, format(insns[start_idx].pc, '02x'), block.length)

                node_attrs = {}
                if not block.covered:
                    node_attrs['fillcolor'] = 'red'
                    node_attrs['style'] = 'filled'

                if start_idx in self.entry_to_method:
                    node_attrs['fillcolor'] = 'yellow'
                    node_attrs['style'] = 'filled'

                graph.node(name=str(start_idx), label=label, **node_attrs)

                if block.default is not None:
                    graph.edge(str(start_idx), str(block.default), color='black')

        return graph


class CFGBlock:

    EXIT = 0xfffffffe
    ERROR = 0xffffffff

    def __init__(self, start_idx, cfg):
        self.start_idx = start_idx
        self.cfg = cfg
        self.length = None
        self.last_insn = None

        self.branch = None
        self.default = None
        self.parents = set()

        self.dep_branches = set()
        self.dep_defaults = set()
        self.dep_parents = set()

        self.covered = False


    def add_block_to_method(self, method):
        method.blocks.add(self.start_idx)

        for i in range(self.start_idx, self.start_idx + self.length):
            method.insns.add(i)
            insn = self.cfg.contract.insns[i]
            method.bow[insn.op] += 1

        if self.default is not None and self.default not in method.blocks:
            self.cfg.blocks[self.default].add_block_to_method(method)

        if self.branch is not None and self.branch not in method.blocks:
            self.cfg.blocks[self.branch].add_block_to_method(method)

        for dep_default in self.dep_defaults:
            if dep_default not in method.blocks:
                self.cfg.blocks[dep_default].add_block_to_method(method)

        for dep_branch in self.dep_branches:
            if dep_branch not in method.blocks:
                self.cfg.blocks[dep_branch].add_block_to_method(method)


    def storage_args(self, method, result, visited_blocks):
        if self.start_idx in visited_blocks and visited_blocks[self.start_idx] >= BLOCK_VISIT_LIMIT:
            return

        if self.start_idx not in visited_blocks:
            visited_blocks[self.start_idx] = 0
        visited_blocks[self.start_idx] += 1

        insns = self.cfg.contract.insns

        for i in range(self.start_idx, self.start_idx + self.length):
            insn = insns[i]

            if insn.op == SSTORE:
                if 'SSTORE' not in result:
                    result['SSTORE'] = set()
                res = self.trace_arg(i, -1 - STACK_CHANGES[insn.op], True, insn.op, method, 0, dict())
                if res is not None:
                    result['SSTORE'].add(res)
                # print('{} {}'.format(res, insn))
            elif insn.op == SLOAD:
                if 'SLOAD' not in result:
                    result['SLOAD'] = set()
                res = self.trace_arg(i, -1 - STACK_CHANGES[insn.op], True, insn.op, method, 0, dict())
                if res is not None:
                    result['SLOAD'].add(res)
                # print('{} {}'.format(res, insn))

        if self.branch is not None and self.branch in method.blocks:
            self.cfg.blocks[self.branch].storage_args(method, result, visited_blocks)

        if self.default is not None and self.default in method.blocks:
            self.cfg.blocks[self.default].storage_args(method, result, visited_blocks)

        # TODO handle dependent blocks better by abstrcting block trace of states
        for dep_branch in self.dep_branches:
            if dep_branch in method.blocks:
                self.cfg.blocks[dep_branch].storage_args(method, result, visited_blocks)

        for dep_default in self.dep_defaults:
            if dep_default in method.blocks:
                self.cfg.blocks[dep_default].storage_args(method, result, visited_blocks)


    def trace_mstore_arg(self, insn_idx, offset, method, depth, visited_blocks):
        if self.start_idx not in visited_blocks:
            visited_blocks[self.start_idx] = 0
        visited_blocks[self.start_idx] += 1

        insns = self.cfg.contract.insns
        for i in range(insn_idx, self.start_idx - 1, -1):
            insn = insns[i]
            if insn.op in (MSTORE, MSTORE8):
                mstore_offset = self.trace_arg(i, -1 - STACK_CHANGES[insn.op], True, insn.op, method, depth+1, dict())
                mstore_value = self.trace_arg(i, -2 - STACK_CHANGES[insn.op], True, insn.op, method, depth+1, dict())
                if mstore_offset is not None and offset == mstore_offset:
                    # print('{} {} {}'.format(insn, format(mstore_offset, '02x'), format(mstore_value, '02x')))
                    return mstore_value

        results = set()

        for parent in self.parents:
            if not visited_blocks[parent] >= BLOCK_VISIT_LIMIT and parent in method.blocks:
                parent_block = self.cfg.blocks[parent]
                result = self.trace_mstore_arg(parent_block.start_idx + parent_block.end_idx - 1, offset, method, depth+1, visited_blocks)
                if result is not None:
                    results.add(result)

        for parent in self.dep_parents:
            if not visited_blocks[parent] >= BLOCK_VISIT_LIMIT and parent in method.blocks:
                parent_block = self.cfg.blocks[parent]
                result = self.trace_mstore_arg(parent_block.start_idx + parent_block.end_idx - 1, offset, method, depth+1, visited_blocks)
                if result is not None:
                    results.add(result)

        if len(results) == 1:
            return list(results)[0]
        else:
            return None


    def trace_arg(self, insn_idx, stack_pos, is_start, target_op, method, depth, visited_blocks):
        if depth > 500:
            return None

        insns = self.cfg.contract.insns
        insn = insns[insn_idx]
        op = insn.op
        if self.start_idx not in visited_blocks:
            visited_blocks[self.start_idx] = 0
        visited_blocks[self.start_idx] += 1

        # print(insn)
        # print(stack_pos)

        old_stack_pos = stack_pos

        if is_push(op):
            if stack_pos == -1 and not is_start:
                return insn.arg
            else:
                stack_pos += STACK_CHANGES[op]
        elif op < 0x10:
            if op == STOP:
                return None
            elif op == ADD:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return (s0 + s1) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == MUL:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return (s0 * s1) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SUB:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return (s0 - s1) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == DIV:
                if stack_pos == -1 and not is_start:
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s1 is not None and s1 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return s0 // s1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SDIV:
                if stack_pos == -1 and not is_start:
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s1 is not None and s1 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    s0, s1 = to_signed(s0), to_signed(s1)
                    return (abs(s0) // abs(s1) * (-1 if s0 * s1 < 0 else 1)) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == MOD:
                if stack_pos == -1 and not is_start:
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s1 is not None and s1 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return s0 % s1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SMOD:
                if stack_pos == -1 and not is_start:
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s1 is not None and s1 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    s0, s1 = to_signed(s0), to_signed(s1)
                    return (abs(s0) % abs(s1) * (-1 if s0 < 0 else 1)) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == ADDMOD:
                if stack_pos == -1 and not is_start:
                    s2 = self.trace_arg(insn_idx, -3 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s2 is not None and s2 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None or s2 is None:
                        return None
                    return (s0 + s1) % s2
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == MULMOD:
                if stack_pos == -1 and not is_start:
                    s2 = self.trace_arg(insn_idx, -3 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s2 is not None and s2 == 0:
                        return 0
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None or s2 is None:
                        return None
                    return (s0 * s1) % s2
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == EXP:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return pow(s0, s1, TT256)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SIGNEXTEND:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    if s0 <= 31:
                        testbit = s0 * 8 + 7
                        if s1 & (1 << testbit):
                            res = s1 | (TT256 - (1 << testbit))
                        else:
                            res = s1 & ((1 << testbit) - 1)
                    else:
                        res = s1
                    return res
                else:
                    stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif op < 0x20:
            if op == LT:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 < s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == GT:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 > s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SLT:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    s0, s1 = to_signed(s0), to_signed(s1)
                    return int(s0 < s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SGT:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    s0, s1 = to_signed(s0), to_signed(s1)
                    return int(s0 > s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == EQ:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 == s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == ISZERO:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None:
                        return None
                    return int(s0 == 0)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == AND:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 & s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == OR:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 | s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == XOR:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return int(s0 ^ s1)
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == NOT:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None:
                        return None
                    return TT256M1 - s0
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == BYTE:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is not None and s0 >= 32:
                        return 0
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return (s1 // 256 ** (31 - s0)) % 256
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SHL:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is not None and s0 >= 256:
                        return 0
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return numpy.left_shift(s1, s0) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SHR:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is not None and s0 >= 256:
                        return 0
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    return numpy.right_shift(s1, s0) & TT256M1
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SAR:
                if stack_pos == -1 and not is_start:
                    s0 = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    s1 = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                    if s0 is None or s1 is None:
                        return None
                    if s0 >= 256:
                        if s1 > 0:
                            res = 0
                        else:
                            res = -1
                    else:
                        res = (s1 >> s0) & TT256M1
                    return res
                else:
                    stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif op < 0x40:
            if op == SHA3:
                if stack_pos == -1 and not is_start:
                    if target_op in (SLOAD, SSTORE, MSTORE, MSTORE8):
                        offset = self.trace_arg(insn_idx, -1 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                        length = self.trace_arg(insn_idx, -2 - STACK_CHANGES[op], True, insn.op, method, depth+1, dict())
                        if offset is not None and length is not None and length == 0x40:
                            # print('{} {} {}'.format(insn, format(offset, '02x'), format(length, '02x')))
                            field_mem_off = offset + 0x20
                            result = self.trace_mstore_arg(insn_idx, field_mem_off, method, depth+1, dict())
                            return result
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op in (ADDRESS, ORIGIN, CALLER, CALLVALUE, CODESIZE, GASPRICE, RETURNDATASIZE):
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op in (BALANCE, EXTCODESIZE):
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == CALLDATASIZE:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == CALLDATALOAD:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op in (CALLDATACOPY, CODECOPY, RETURNDATACOPY):
                stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif op < 0x50:
            if op == BLOCKHASH:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op in (COINBASE, TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT):
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif op < 0x60:
            if op == POP:
                stack_pos += STACK_CHANGES[op]
            elif op == MLOAD:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == MSTORE:
                stack_pos += STACK_CHANGES[op]
            elif op == MSTORE8:
                stack_pos += STACK_CHANGES[op]
            elif op == SLOAD:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == SSTORE:
                stack_pos += STACK_CHANGES[op]
            elif op == JUMP:
                stack_pos += STACK_CHANGES[op]
            elif op == JUMPI:
                stack_pos += STACK_CHANGES[op]
            elif op == PC:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == MSIZE:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == GAS:
                if stack_pos == -1 and not is_start:
                    return None
                else:
                    stack_pos += STACK_CHANGES[op]
            elif op == JUMPDEST:
                stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif is_dup(op):
            if stack_pos == -1:
                stack_pos = 0x7f - op
            else:
                stack_pos += STACK_CHANGES[op]
        elif is_swap(op):
            if stack_pos == -1:
                stack_pos = 0x8e - op
            elif stack_pos == 0x8e - op:
                stack_pos = -1
            else:
                stack_pos += STACK_CHANGES[op]
        elif 0xa0 <= op <= 0xa4: # LOG
            if op == LOG0:
                stack_pos += STACK_CHANGES[op]
            elif op == LOG1:
                stack_pos += STACK_CHANGES[op]
            elif op == LOG2:
                stack_pos += STACK_CHANGES[op]
            elif op == LOG3:
                stack_pos += STACK_CHANGES[op]
            elif op == LOG4:
                stack_pos += STACK_CHANGES[op]
            else:
                assert False
        elif op == CREATE:
            if stack_pos == -1 and not is_start:
                return None
            else:
                stack_pos += STACK_CHANGES[op]
        elif op in (CALL, CALLCODE):
            if stack_pos == -1 and not is_start:
                return None
            else:
                stack_pos += STACK_CHANGES[op]
        elif op == RETURN:
            return None
        elif op in (DELEGATECALL, STATICCALL):
            if stack_pos == -1 and not is_start:
                return None
            else:
                stack_pos += STACK_CHANGES[op]
        elif op == REVERT:
            return None
        elif op == INVALID:
            return None
        elif op == SELFDESTRUCT:
            return None
        else:
            assert False, 'unsuppored opcode {}'.format(op)

        new_stack_pos = stack_pos

        if new_stack_pos - old_stack_pos != STACK_CHANGES[op] and not is_dup(op) and not is_swap(op):
            assert False

        if insn_idx > self.start_idx:
            return self.trace_arg(insn_idx-1, stack_pos, False, target_op, method, depth+1, visited_blocks)
        elif insn_idx == self.start_idx:
            results = set()

            for parent in self.parents:
                if not (parent in visited_blocks and visited_blocks[parent] >= BLOCK_VISIT_LIMIT) and parent in method.blocks:
                    parent_block = self.cfg.blocks[parent]
                    result = parent_block.trace_arg(parent_block.start_idx + parent_block.length - 1, stack_pos, False, target_op, method, depth+1, visited_blocks)
                    if result is not None:
                        results.add(result)

            for parent in self.dep_parents:
                if not (parent in visited_blocks and visited_blocks[parent] >= BLOCK_VISIT_LIMIT) and parent in method.blocks:
                    parent_block = self.cfg.blocks[parent]
                    result = parent_block.trace_arg(parent_block.start_idx + parent_block.length - 1, stack_pos, False, target_op, method, depth+1, visited_blocks)
                    if result is not None:
                        results.add(result)

            if len(results) == 1:
                return list(results)[0]
            else:
                return None
        else:
            return None

        return None