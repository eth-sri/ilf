from copy import copy, deepcopy
import logging
import inspect
import re
import z3
from collections import Counter
from ilf.symbolic import utils
from ilf.symbolic.exceptions import SVMRuntimeError
from ilf.symbolic.solidity import asm
from ilf.symbolic.solidity import solidity_utils
from ilf.symbolic.symbolic import constraints
from ilf.symbolic.symbolic import svm_utils
from ilf.symbolic.symbolic.account import Account, AccountType
from ilf.symbolic.symbolic.environment import CalldataType, Environment
from ilf.symbolic.symbolic.global_state import GlobalState
from ilf.symbolic.symbolic.world_state import WorldState, WorldStateStatus
from ilf.symbolic.symbolic.svm_utils import TT256, TT256M1, make_trace
from ilf.symbolic.utils import BColors, ADDRESS_ARG_TAG

SYMBOLIC_CALL_RETURNDATA_SIZE_BYTES = 32
SHA_EMPTY_ARGS_VALUE = 2**255 - 0x12345

class Executor:
    def __init__(self, svm):
        self.svm = svm
        self.op_to_stack_len = {}

    def execute_gstate(self, gstate):
        next_gstates = []
        environment = gstate.environment
        active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
        active_account.balance += environment.callvalue

        while not gstate.halt:
            eval_gstates = self.evaluate(gstate)
            eval_gstates = eval_gstates if eval_gstates is not None else []
            next_gstates.extend(eval_gstates)

        if gstate.wstate.trace is None:
            gstate.wstate.trace = make_trace(active_account, svm_utils.FALLBACK_FUNC_NAME)

        # if gstate.wstate.gen == 0:
            # gstate.wstate.trace = make_trace(active_account, svm_utils.INIT_FUNC_NAME)

        return next_gstates

    def pre_evaluate(self, gstate):
        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        instr_address = instr['address']
        active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
        gstate.pc_trace.append((active_account.contract.name, instr_address))


    def evaluate(self, gstate):
        stack_len_start = len(gstate.mstate.stack)
        self.pre_evaluate(gstate)
        if gstate.halt:
            return

        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        instr_address = instr['address']


        op = instr['opcode']
        match = re.match(r'^(PUSH|DUP|LOG|SWAP)\d{1,2}', op)
        op = match.group(1) if match else op
        eval_func = getattr(self, op, None)
        if eval_func is None:
            raise SVMRuntimeError(f'op evaluator not found: {op}')

        active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
        current_func = '?'  if gstate.wstate.trace is None else gstate.wstate.trace
        arg = instr.get('argument', '')
        arg = (arg[0:10] + '..') if len(arg) > 12 else arg.ljust(12)
        logging.debug(f'{BColors.BLUE}{BColors.BOLD}OP{BColors.ENDC} '
                      f'{op.ljust(12)}\t'
                      f'{arg},\t'
                      f'addr={instr_address},\t'
                      f'pc={gstate.mstate.pc},\t'
                      f'contract={active_account.contract.name}\t'
                      f'func={current_func}\t'
                      f'sender={gstate.environment.sender}')
        arglist = inspect.getargspec(eval_func).args
        try:
            stack_args = [gstate.mstate.stack.pop() for arg in arglist[2:]]
            res = eval_func(gstate, *stack_args)
            gstate.mstate.pc += 1
            stack_len_stop = len(gstate.mstate.stack)
            self.op_to_stack_len[op] = (stack_len_stop - stack_len_start)
            return res
        except Exception as e:
            s = z3.Solver()
            s.add(gstate.wstate.constraints)
            if s.check() == z3.sat:
                raise e

    def PUSH(self, gstate):
        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        value = z3.BitVecVal(int(instr['argument'][2:], 16), 256)
        gstate.mstate.stack.append(value)

    def DUP(self, gstate):
        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        depth = int(instr['opcode'][3:])
        gstate.mstate.stack.append(gstate.mstate.stack[-depth])

    def SWAP(self, gstate):
        mstate = gstate.mstate
        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        depth = int(instr['opcode'][4:])
        temp = mstate.stack[-depth - 1]
        mstate.stack[-depth - 1] = mstate.stack[-1]
        mstate.stack[-1] = temp

    def POP(self, gstate, a):
        pass

    def AND(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a & b)

    def OR(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a | b)

    def XOR(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a ^ b)

    def NOT(self, gstate, a):
        gstate.mstate.stack.append(TT256M1 - a)

    def BYTE(self, gstate, byte_index, word):
        bits_low = (31 - svm_utils.get_concrete_int(byte_index)) * 8
        result = svm_utils.zpad_bv_right(z3.Extract(bits_low + 7, bits_low, word), 256)
        gstate.mstate.stack.append(result)

    def ADD(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a + b)

    def SUB(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a - b)

    def MUL(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        if self.svm.abs_mul and not svm_utils.is_bv_concrete(a) and not svm_utils.is_bv_concrete(b):
            abs_bv = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.MUL,
                                                              gstate.wstate.gen,
                                                              unique=True)
            gstate.mstate.stack.append(abs_bv)
        elif svm_utils.is_bv_pow2(a) or svm_utils.is_bv_pow2(b):
            if svm_utils.is_bv_pow2(b):
                a, b = b, a
            a = svm_utils.get_concrete_int(a)
            i = 0
            while a != (1 << i):
                i += 1
            gstate.mstate.stack.append(z3.simplify(b << i))
        else:
            gstate.mstate.stack.append(a * b)

    def DIV(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        if self.svm.abs_div and not svm_utils.is_bv_concrete(a) and not svm_utils.is_bv_concrete(b):
            abs_bv = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.DIV,
                                                              gstate.wstate.gen,
                                                              unique=True)
            gstate.mstate.stack.append(abs_bv)
        else:
            gstate.mstate.stack.append(z3.UDiv(a, b))

    def MOD(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(0 if b == 0 else z3.URem(a, b))

    def SDIV(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(a / b)

    def SMOD(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        gstate.mstate.stack.append(0 if b == 0 else a % b)

    def ADDMOD(self, gstate, a, b, c):
        a, b, c = map(svm_utils.convert_to_bitvec, (a, b, c))
        gstate.mstate.stack.append((a + b) % c)

    def MULMOD(self, gstate, a, b, c):
        a, b, c = map(svm_utils.convert_to_bitvec, (a, b, c))
        gstate.mstate.stack.append((a * b) % c if c else 0)

    def EXP(self, gstate, base, exponent):
        base, exponent = map(svm_utils.convert_to_bitvec, (base, exponent))
        if svm_utils.is_bv_concrete(base) and svm_utils.is_bv_concrete(exponent):
            base = svm_utils.get_concrete_int(base)
            exponent = svm_utils.get_concrete_int(exponent)
            value = pow(base, exponent, 2**256)
            gstate.mstate.stack.append(z3.BitVecVal(value, 256))
        else:
            active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
            gstate.mstate.stack.append(self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.EXP,
                                                                                gstate.wstate.gen,
                                                                                unique=True,
                                                                                acc=active_account.id))
    def SIGNEXTEND(self, gstate, a, b):
        if svm_utils.is_bv_concrete(a) and svm_utils.is_bv_concrete(b):
            a = svm_utils.get_concrete_int(a)
            b = svm_utils.get_concrete_int(b)
            if a <= 31:
                testbit = a * 8 + 7
                if b & (1 << testbit):
                    gstate.mstate.stack.append(b | (TT256 - (1 << testbit)))
                else:
                    gstate.mstate.stack.append(b & ((1 << testbit) - 1))
            else:
                gstate.mstate.stack.append(b)
        else:
            raise SVMRuntimeError('SIGNEXTEND error')

    def LT(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        exp = z3.ULT(a, b)
        gstate.mstate.stack.append(exp)

    def GT(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        exp = z3.UGT(a, b)
        gstate.mstate.stack.append(exp)

    def SLT(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        exp = (a < b)
        gstate.mstate.stack.append(exp)

    def SGT(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        exp = (a > b)
        gstate.mstate.stack.append(exp)

    def EQ(self, gstate, a, b):
        a, b = map(svm_utils.convert_to_bitvec, (a, b))
        exp = (a == b)
        gstate.mstate.stack.append(exp)

    def ISZERO(self, gstate, a):
        a = svm_utils.convert_to_bitvec(a)
        gstate.mstate.stack.append(a == 0)

    def CALLVALUE(self, gstate):
        gstate.mstate.stack.append(gstate.environment.callvalue)

    def CALLDATALOAD(self, gstate, index):
        logging.debug('CALLDATA index:' + str(index))
        if gstate.environment.calldata is None:
            raise SVMRuntimeError('CALLDATA is not set')
        index_concrete = svm_utils.is_bv_concrete(index)
        if gstate.environment.calldata_type == CalldataType.UNDEFINED:
            data_bytes = []
            # for i in range(32):
                # label = 'calldata_{}_{}'.format(wstate.gen, i + svm_utils.get_concrete_int(index))
                # data_bytes.append(z3.BitVec(label, 8))
            # data = z3.Concat(data_bytes)
            if index_concrete:
                data = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                            gstate.wstate.gen,
                                                            index=svm_utils.get_concrete_int(index))
                gstate.mstate.stack.append(data)
            else:
                data = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                            gstate.wstate.gen,
                                                            unique=True)
                gstate.mstate.stack.append(data)

        elif gstate.environment.calldata_type == CalldataType.DEFINED:
            assert gstate.environment.calldata.sort().name() == 'bv', 'CALLDATA sort mismatch'
            index = svm_utils.get_concrete_int(z3.simplify(index))
            calldatasize = gstate.environment.calldata.size()
            assert index < calldatasize
            calldata_start = calldatasize-index*8
            calldata_stop = max(0, calldata_start - 256)
            calldata = z3.Extract(calldata_start-1, calldata_stop, gstate.environment.calldata)
            calldata = svm_utils.zpad_bv_left(calldata, 256)
            gstate.mstate.stack.append(calldata)
        else:
            raise SVMRuntimeError('Unknown calldata type')

    def CALLDATACOPY(self, gstate, dest_offset, offset, length):
        length_concrete = svm_utils.is_bv_concrete(length)
        if not length_concrete:
            logging.warning('Symbolic calldata size')
            length = z3.BitVecVal(64, 256)
        if gstate.environment.calldata_type == CalldataType.UNDEFINED:
            length = svm_utils.get_concrete_int(length)
            if svm_utils.is_bv_concrete(offset):
                offset = svm_utils.get_concrete_int(offset)
                for i in range(length):
                    data_word = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                                         gstate.wstate.gen,
                                                                         index=offset+(i//32))
                    slot = i % 32
                    data_bytes = svm_utils.split_bv_into_bytes(data_word)
                    data = data_bytes[slot]
                    gstate.mstate.memory = z3.Store(gstate.mstate.memory, dest_offset+i, data)
            else:
                for i in range(length):
                    data = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA,
                                                                    gstate.wstate.gen,
                                                                    unique=True,
                                                                    bv_size=8)
                    gstate.mstate.memory = z3.Store(gstate.mstate.memory, dest_offset+i, data)
        elif gstate.environment.calldata_type == CalldataType.DEFINED:
            length = svm_utils.get_concrete_int(length)
            offset_concrete = svm_utils.is_bv_concrete(offset)
            calldata_bytes = svm_utils.split_bv_into_bytes(gstate.environment.calldata)
            offset_concrete = svm_utils.is_bv_concrete(offset)
            for i in range(length):
                gstate.mstate.memory = z3.Store(gstate.mstate.memory, dest_offset+i, calldata_bytes[offset_concrete+i])
        else:
            raise SVMRuntimeError('Unknown calldata type')


    def CALLDATASIZE(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]

        if gstate.environment.calldata_type == CalldataType.UNDEFINED:
            gstate.mstate.stack.append(self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATASIZE,
                                                                                gstate.wstate.gen,
                                                                                acc=active_account.id))
        elif gstate.environment.calldata_type == CalldataType.DEFINED:
            if gstate.environment.calldata is not None:
                gstate.mstate.stack.append(z3.BitVecVal(gstate.environment.calldata.size() // 8, 256))
            else:
                gstate.mstate.stack.append(z3.BitVecVal(0, 256))
        else:
            raise SVMRuntimeError('Unknown calldata type')

    def STOP(self, gstate):
        gstate.exit_code = 1
        gstate.halt = True
        return [gstate]

    def ADDRESS(self, gstate):
        gstate.mstate.stack.append(z3.BitVecVal(gstate.environment.active_address, 256))

    def BALANCE(self, gstate, address):
        if svm_utils.is_bv_concrete(address) and svm_utils.get_concrete_int(address) in gstate.wstate.address_to_account:
            balance = gstate.wstate.address_to_account[svm_utils.get_concrete_int(address)].balance
            gstate.mstate.stack.append(balance)
            return

        address = str(z3.simplify(address)).replace(' ', '_')
        gstate.mstate.stack.append(self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.BALANCE,
                                                                            gstate.wstate.gen,
                                                                            unique=True))

    def ORIGIN(self, gstate):
        gstate.mstate.stack.append(gstate.environment.origin)

    def CALLER(self, gstate):
        gstate.mstate.stack.append(gstate.environment.sender)

    def CODESIZE(self, gstate):
        gstate.mstate.stack.append(z3.BitVecVal(len(gstate.environment.runtime_bytecode_bytes), 256))

    def SHA3(self, gstate, index, length):
        if svm_utils.is_bv_concrete(index) and svm_utils.is_bv_concrete(length):
            index = svm_utils.get_concrete_int(index)
            length = svm_utils.get_concrete_int(length)
            if length == 0:
                gstate.mstate.stack.append(z3.BitVecVal(SHA_EMPTY_ARGS_VALUE, 256))
                return
            data = z3.simplify(svm_utils.get_memory_data(gstate.mstate.memory, index, length))
            sha_constraints, hash_vector = svm_utils.symbolic_keccak(self, data)
            gstate.wstate.constraints.extend(sha_constraints)
            gstate.mstate.stack.append(hash_vector)
            return
        hash_vector = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.SHA3,
                                                               gstate.wstate.gen,
                                                               unique=True)
        logging.debug('SHA index or len not resolved. Using the symbolic vector')
        gstate.mstate.stack.append(hash_vector)
        return

    def GASPRICE(self, gstate):
        gstate.mstate.stack.append(gstate.environment.gasprice)

    def CODECOPY(self, gstate, dest_offset, offset, length):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]

        offset = svm_utils.get_concrete_int(offset)
        if svm_utils.is_bv_concrete(length):
            length = svm_utils.get_concrete_int(length)
        else:
            length = 0
        assert isinstance(gstate.environment.runtime_bytecode_bytes, list)
        bytecode_bytes = gstate.environment.runtime_bytecode_bytes
        for i in range(length):
            if offset + i < len(bytecode_bytes):
                data_byte = bytecode_bytes[offset+i]
            else:
                data_byte = z3.BitVecVal(0, 8)
            gstate.mstate.memory = z3.Store(gstate.mstate.memory, dest_offset+i, data_byte)
            gstate.mstate.memory_dict[svm_utils.get_concrete_int(dest_offset+i)] = data_byte

        # if offset+length-1 >= len(bytecode_bytes):
            # instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
            # instr_address = instr['address']
            # current_contract = active_account.contract
            # line = solidity_utils.offset_to_line(current_contract.src_code, instr_address, current_contract.src_map)
            # src_code = current_contract.src_code.split('\n')[line].strip()
            # raise SVMRuntimeError('CODECOPY index out of bounds!')

    def EXTCODESIZE(self, gstate, addr):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        extcodesize = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.BLOCKHASH,
                                                             gstate.wstate.gen,
                                                             unique=True,
                                                             acc=active_account.id)
        gstate.mstate.stack.append(extcodesize)

    def EXTCODECOPY(self, gstate, start, s2, size):
        raise SVMRuntimeError('not implemented')

    def BLOCKHASH(self, gstate, blocknumber):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        blockhash = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.BLOCKHASH,
                                                             gstate.wstate.gen,
                                                             acc=active_account.id,
                                                             unique=True)
        gstate.mstate.stack.append(blockhash)

    def COINBASE(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        coinbase = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.COINBASE,
                                                            gstate.wstate.gen,
                                                            acc=active_account.id)
        gstate.mstate.stack.append(coinbase)

    def TIMESTAMP(self, gstate):
        gstate.mstate.stack.append(gstate.environment.timestamp)

    def NUMBER(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        number = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.NUMBER,
                                                          gstate.wstate.gen,
                                                          acc=active_account.id)
        gstate.mstate.stack.append(number)
    def DIFFICULTY(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        difficulty = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.DIFFICULTY,
                                                              gstate.wstate.gen,
                                                              acc=active_account.id)
        gstate.mstate.stack.append(difficulty)
    def GASLIMIT(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        gaslimit = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.GASLIMIT,
                                                            gstate.wstate.gen,
                                                            acc=active_account.id)
        gstate.mstate.stack.append(gaslimit)
    def MLOAD(self, gstate, index):
        value_bytes = []
        for i in range(32):
           value_bytes.append(z3.Select(gstate.mstate.memory, index+i))
        value = z3.simplify(z3.Concat(value_bytes))
        gstate.mstate.stack.append(value)

    def MSTORE(self, gstate, index, value):
        if isinstance(value, z3.BoolRef):
            value = z3.If(value, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
        value_bytes = svm_utils.split_bv_into_bytes(value)
        for i in range(32):
            if svm_utils.is_bv_concrete(index):
                gstate.mstate.memory_dict[svm_utils.get_concrete_int(index)+i] = value_bytes[i]
            gstate.mstate.memory = z3.Store(gstate.mstate.memory, index+i, value_bytes[i])

    def MSTORE8(self, gstate, index, value):
        data = z3.Extract(7, 0, value)
        if svm_utils.is_bv_concrete(index):
            gstate.mstate.memory_dict[svm_utils.get_concrete_int(index)] = data
        gstate.mstate.memory = z3.Store(gstate.mstate.memory, index, data)

    def SLOAD(self, gstate, index):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        data = active_account.storage.load(index)
        gstate.mstate.stack.append(data)

    def SSTORE(self, gstate, index, value):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        active_account.storage.store(index, value)
        gstate.has_storage_changed = True

    def JUMP(self, gstate, jump_addr):
        if not svm_utils.is_bv_concrete(jump_addr):
            if svm_utils.check_wstate_reachable(gstate.wstate, 100):
                logging.warning('JUMP to invalid address')
            return
        # if gstate.pc_addr_to_depth.setdefault((gstate.mstate.pc, jump_addr), 0) < self.svm.max_jump_depth:
        jump_addr = svm_utils.get_concrete_int(jump_addr)
        instr_idx = svm_utils.get_instruction_index(gstate.environment.disassembly.instruction_list, jump_addr)
        if instr_idx is None:
            raise SVMRuntimeError('JUMP to invalid address')
        dest_opcode = gstate.environment.disassembly.instruction_list[instr_idx]['opcode']
        if dest_opcode != 'JUMPDEST':
            raise SVMRuntimeError('JUMP to invalid address')
        gstate.mstate.pc = instr_idx
        return self.execute_gstate(gstate)

    def JUMPI(self, gstate, jump_addr, condition):
        new_gstates = []

        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        mstate = gstate.mstate

        jump_addr = svm_utils.get_concrete_int(jump_addr)
        function_entry = False

        gstate.jump_count += 1
        if (gstate.jump_count % 3 == 0 and
                gstate.jump_count > 20 and
                not svm_utils.check_wstate_reachable(gstate.wstate, 100)):
            gstate.halt = True
            return

        if gstate.pc_addr_to_depth.setdefault((gstate.mstate.pc, jump_addr), 0) >= 3:
            gstate.halt = True
            return
        increase_depth = True
        current_contract = active_account.contract
        line = solidity_utils.offset_to_line(current_contract.src_code, mstate.pc, current_contract.src_map)
        src_code = current_contract.src_code.split('\n')[line].strip()
        concrete_cond = False
        if 'assert' in src_code:
            increase_depth = False
        if type(condition) == bool:
            increase_depth = False
        else:
            simplified_cond = z3.simplify(condition)
            if z3.is_true(simplified_cond) or z3.is_false(simplified_cond):
                concrete_cond = True
                increase_depth = False
        if increase_depth:
            gstate.pc_addr_to_depth[(mstate.pc, jump_addr)] = gstate.pc_addr_to_depth[(mstate.pc, jump_addr)] + 1

        instr_idx = svm_utils.get_instruction_index(gstate.environment.disassembly.instruction_list, jump_addr)
        if instr_idx is None:
            raise SVMRuntimeError('JUMP to invalid address')
        dest_opcode = gstate.environment.disassembly.instruction_list[instr_idx]['opcode']
        if dest_opcode != 'JUMPDEST':
            raise SVMRuntimeError('JUMP to invalid address')

        condition = z3.BoolVal(condition) if isinstance(condition, bool) else condition
        condition = (condition == 0) if isinstance(condition, z3.BitVecRef) else condition
        assert isinstance(condition ,z3.BoolRef), 'Invalid condition types!'

        if not z3.is_false(z3.simplify(condition)):
            true_gstate = copy(gstate)
            true_gstate.mstate.pc = instr_idx
            true_gstate.wstate.constraints.append(condition)
            if true_gstate.wstate.trace is None:
                jump_func_name, jump_func_hash = gstate.environment.extract_func_name_hash(gstate.mstate.pc)
                if jump_func_name:
                    jump_trace = make_trace(active_account, jump_func_name)
                    logging.debug('Entering function %s', jump_trace)
                    true_gstate.wstate.trace = jump_trace
            new_gstates.extend(self.execute_gstate(true_gstate))

        negated_condition = z3.Not(condition)

        if not z3.is_false(z3.simplify(negated_condition)):
            false_gstate = copy(gstate)
            false_gstate.mstate.pc += 1
            false_gstate.wstate.constraints.append(negated_condition)
            new_gstates.extend(self.execute_gstate(false_gstate))

        gstate.halt = True
        return new_gstates

    def PC(self, gstate):
        gstate.mstate.stack.append(gstate.mstate.pc - 1)

    def MSIZE(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        gstate.mstate.stack.append(self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.MSIZE,
                                                                            gstate.wstate.gen,
                                                                            acc=active_account.id))

    def GAS(self, gstate):
        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]
        gstate.mstate.stack.append(self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.GAS,
                                                                           gstate.wstate.gen,
                                                                           acc=active_account.id))
    def LOG(self, gstate, address, size):
        instr = gstate.environment.disassembly.instruction_list[gstate.mstate.pc]
        depth = int(instr['opcode'][3:])
        log_data = [gstate.mstate.stack.pop() for _ in range(depth)]

    def CREATE(self, gstate, balance, offset, length):
        new_gstates = []

        active_address = gstate.environment.active_address
        active_account = gstate.wstate.address_to_account[active_address]

        if gstate.wstate.gen != 0:
            raise SVMRuntimeError('Dynamic contract creation is not supported')

        offset = svm_utils.get_concrete_int(offset)
        length = svm_utils.get_concrete_int(length)
        create_contract_bytes = []
        concrete_prefix_bytes = []
        concrete_prefix_stop = False
        for index in range(0, length):
            memory_byte = gstate.mstate.memory_dict[index+offset]
            memory_byte = svm_utils.get_concrete_int(memory_byte) if svm_utils.is_bv_concrete(memory_byte) else memory_byte
            create_contract_bytes.append(memory_byte)
            concrete_prefix_stop = concrete_prefix_stop or not isinstance(memory_byte, int)
            if not concrete_prefix_stop: concrete_prefix_bytes.append(memory_byte)
        found_swarmhashes = asm.find_swarmhashes(bytes(concrete_prefix_bytes))
        if not len(found_swarmhashes):
            raise SVMRuntimeError('CREATE found no swarmhashes in bytecode')
        create_contract = self.svm.swarm_hash_tuple_to_contract[tuple(found_swarmhashes)]
        create_address = utils.get_next_contract_address()
        create_account = Account(create_address, create_contract, balance=balance)
        logging.debug(f'CREATE created contract {create_account.contract.name}')
        create_enviroment = Environment(active_address=create_address,
                                        sender=z3.BitVecVal(active_account.address, 256),
                                        calldata=[],
                                        gasprice=gstate.environment.gasprice,
                                        callvalue=balance,
                                        origin=gstate.environment.origin,
                                        calldata_type=CalldataType.UNDEFINED,
                                        disassembly=create_account.contract.creation_disassembly,
                                        runtime_bytecode_bytes=create_contract_bytes,
                                        timestamp=gstate.environment.timestamp)
        child_wstate = copy(gstate.wstate)
        child_wstate.address_to_account[create_address] = create_account
        create_gstate = GlobalState(child_wstate, create_enviroment)
        create_gstate.has_storage_changed = True
        intermediate_gstates = self.execute_gstate(create_gstate)
        if len(intermediate_gstates) == 0:
            raise SVMRuntimeError('CREATE has no feasible blocks')
        for new_gstate in intermediate_gstates:
            new_gstate.mstate = deepcopy(gstate.mstate)
            new_gstate.mstate.stack.append(z3.BitVecVal(create_address, 256))
            new_gstate.pc_addr_to_depth = copy(gstate.pc_addr_to_depth)
            new_gstate.mstate.pc += 1
            new_gstate.halt = False
            new_gstate.environment = gstate.environment
            new_gstates.extend(self.execute_gstate(new_gstate))
        logging.debug(f'returning FROM CREATE contract {create_account.contract.name}')
        gstate.halt = True
        return new_gstates

    def _CALL(self, gstate, gas, to, value, meminstart, meminsize, outmemstart, memoutsize):
        new_gstates = []

        active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
        environment = gstate.environment
        mstate = gstate.mstate
        wstate = gstate.wstate

        active_account.balance -= value

        if z3.is_true(z3.simplify(to == 0x0)):
            mstate.stack.append(z3.BitVecVal(0, 256))
        elif z3.is_true(z3.simplify(to == 0x1)):
           raise Exception('not implemented')
        elif z3.is_true(z3.simplify(to == 0x2)):
           raise Exception('not implemented')
        elif svm_utils.is_bv_concrete(meminsize):
            meminsize = svm_utils.get_concrete_int(meminsize)
            gstate.return_data = None
            callee_calldata = None
            in_data_bytes = [z3.Select(mstate.memory, meminstart+i) for i in range(meminsize)] if meminsize else []
            if meminsize != 0:
                callee_calldata = z3.Concat(in_data_bytes) if len(in_data_bytes) > 1 else in_data_bytes[0]
            for callee_address, callee_account in wstate.address_to_account.items():
                callee_constraint = z3.simplify(to == callee_address)
                if z3.is_false(callee_constraint):
                    continue
                if callee_address == active_account.address:
                    if svm_utils.is_bv_concrete(gstate.environment.sender) and\
                             svm_utils.get_concrete_int(gstate.environment.sender) == callee_address:
                        gstate.halt = True
                        return []
                callee_environment = Environment(active_address=callee_address,
                                                 sender=z3.BitVecVal(active_account.address, 256),
                                                 calldata=callee_calldata,
                                                 gasprice=environment.gasprice,
                                                 callvalue=value,
                                                 origin=environment.origin,
                                                 calldata_type=CalldataType.DEFINED,
                                                 disassembly=callee_account.contract.disassembly,
                                                 runtime_bytecode_bytes=list(callee_account.contract.disassembly.bytecode),
                                                 timestamp=environment.timestamp)
                child_wstate = copy(wstate)
                child_wstate.constraints.append(to == callee_address)
                if not svm_utils.check_wstate_reachable(child_wstate, 100):
                    continue

                child_gstate = GlobalState(child_wstate, callee_environment)
                intermediate_gstates = self.execute_gstate(child_gstate)
                for new_gstate in intermediate_gstates:
                    new_gstate.mstate = deepcopy(gstate.mstate)
                    new_gstate.mstate.stack.append(new_gstate.exit_code)
                    new_gstate.exit_code = None
                    new_gstate.pc_addr_to_depth = copy(gstate.pc_addr_to_depth)
                    new_gstate.mstate.pc += 1
                    new_gstate.halt = False
                    new_gstate.environment = gstate.environment
                    new_gstate.has_storage_changed |= gstate.has_storage_changed
                    if new_gstate.return_data is not None:
                        return_data = new_gstate.return_data
                        memoutsize = svm_utils.get_concrete_int(memoutsize)
                        outmemstart = svm_utils.get_concrete_int(outmemstart)
                        return_data_bytes = svm_utils.split_bv_into_bytes(return_data)
                        for i in range(min(return_data.size()//8, memoutsize)):
                            return_byte = return_data_bytes[i]
                            new_gstate.mstate.memory = z3.Store(new_gstate.mstate.memory, i + outmemstart, return_byte)
                    new_gstates.extend(self.execute_gstate(new_gstate))
        ret = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.RETURN_VALUE,
                                                       wstate.gen,
                                                       unique=True,
                                                       acc=active_account.id)
        gstate.return_data = None
        to_constraint = z3.Or([to == p for p in self.svm.possible_caller_addresses])
        wstate.constraints.append(to_constraint)
        wstate.has_symbolic_calls = True
        return_data = self.svm.sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.RETURNDATA,
                                                               wstate.gen,
                                                               unique=True,
                                                               bv_size=SYMBOLIC_CALL_RETURNDATA_SIZE_BYTES*8,
                                                               acc=active_account.id)

        gstate.return_data = return_data
        concrete_memoutsize = svm_utils.get_concrete_int(memoutsize)
        if concrete_memoutsize != 0:
            return_data_bytes = svm_utils.split_bv_into_bytes(return_data)
            memoutsize = svm_utils.get_concrete_int(memoutsize)
            for i in range(min(return_data.size()//8, memoutsize)):
                return_byte = return_data_bytes[i]
                mstate.memory = z3.Store(mstate.memory, i+outmemstart, return_byte)
        gstate.has_storage_changed = True
        mstate.stack.append(ret)
        return new_gstates

    def CALL(self, gstate, gas, to, value, meminstart, meminsize, outmemstart, memoutsize):
        self._CALL(gstate, gas, to, value, meminstart, meminsize, outmemstart, memoutsize)

    def CALLCODE(self, gstate, gas, to, value, meminstart, meminsize, outmemstart, memoutsize):
        raise SVMRuntimeError('CALLCODE not implemented')

    def STATICCALL(self, gstate, gas, to, meminstart, meminsize, outmemstart, memoutsize):
        self._CALL(gstate, gas, to, 0, meminstart, meminsize, outmemstart, memoutsize)

    def DELEGATECALL(self, gstate, gas, to, meminstart, meminsize, outmemstart, memoutsize):
        value = 0
        raise SVMRuntimeError('DELEGATECALL not implemented')

    def RETURN(self, gstate, offset, length):
        if not svm_utils.is_bv_concrete(length):
            # raise SVMRuntimeError('Non concrete return length')
            length = 32
        length = svm_utils.get_concrete_int(length)
        return_data_bytes = [z3.Select(gstate.mstate.memory, i+offset) for i in range(length)]
        return_data = z3.Concat(return_data_bytes)
        assert return_data.size() == length*8
        gstate.return_data = return_data
        gstate.exit_code = 1
        gstate.halt = True
        return [gstate]

    def SUICIDE(self, gstate, to):
        gstate.halt = True
        if svm_utils.is_bv_concrete(to) and svm_utils.get_concrete_int(to) in gstate.wstate.address_to_account:
            active_account = gstate.wstate.address_to_account[gstate.environment.active_address]
            gstate.wstate.address_to_account[to].balance += active_account.balance
            active_account.balance = 0
        return [gstate]

    def REVERT(self, gstate):
        gstate.exit_code = 0
        gstate.halt = True
        return [gstate]

    def ASSERT_FAIL(self, gstate):
        gstate.exit_code = 0
        gstate.halt = True
        return [gstate]

    def JUMPDEST(self, gstate):
        pass

    def RETURNDATASIZE(self, gstate):
        ret_data_size = gstate.return_data.size() // 8 if gstate.return_data is not None else 0
        gstate.mstate.stack.append(z3.BitVecVal(ret_data_size, 256))

    def RETURNDATACOPY(self, gstate, dest_offset, offset, length):
        offset = svm_utils.get_concrete_int(offset)
        length = svm_utils.get_concrete_int(length)
        if length == 0:
            return
        return_data_bytes = svm_utils.split_bv_into_bytes(gstate.return_data)
        for i in range(length):
            gstate.mstate.memory = z3.Store(gstate.mstate.memory, dest_offset+i, return_data_bytes[offset+i])
