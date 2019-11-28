from enum import Enum
from ilf.symbolic.symbolic import svm_utils
import z3

# format of constraint: constraintTypeString_tx_accId_additionalData
# ConstraintType.VALUE = ('constraint_type_string', num_of_args_of_additional_data)
class ConstraintType(Enum):
    BYTE = 'byte'
    EXP = 'exp'
    ENTRY_ACCOUNT = 'entry_account'
    CALLDATA_ARRAY = 'calldata_array'
    CALLDATA = 'calldata'
    CALLDATALOAD = 'calldataload'
    CALLDATASIZE = 'calldatasize'
    CALLDATACOPY = 'calldatacpy'
    CALLER = 'caller'
    CALLVALUE = 'callvalue'
    BALANCE = 'balance'
    SHA3 = 'sha'
    GASPRICE = 'gasprice'
    CODECOPY = 'codecopy'
    EXTCODESIZE = 'extcodesize'
    BLOCKHASH = 'blockhash'
    COINBASE = 'coinbase'
    TIMESTAMP = 'timestamp'
    NUMBER = 'block_number'
    DIFFICULTY = 'block_difficulty'
    GASLIMIT = 'block_gaslimit'
    MSIZE = 'msize'
    GAS = 'gas'
    CALL = 'call_return'
    ORIGIN = 'origin'
    SLOAD = 'sload'
    RETURN = 'return'
    RETURN_VALUE = 'retval'
    RETURNDATA = 'returndata'
    OUTPUT = 'output'
    SYMBOLIC = 'symbolic'
    CREATED_CONTRACT = 'created_contract'
    EXPLORE_SHA = 'explore_sha'
    MUL = 'mul'
    DIV = 'div'



class SymbolicBitvecGenerator():
    def __init__(self):
        self.unique_vector_name_counter = {}

    def get_sym_bitvec(self, constraint_type, gen, bv_size=256, unique=False, **kwargs):
        vector_name = ConstraintType[constraint_type.name].value
        label_template = vector_name + '_gen{}'
        for k in kwargs:
            label_template += '_' + k + '{' + k + '}'
        label = label_template.format(gen, **kwargs)
        if unique:
            unique_id = self.unique_vector_name_counter.get(vector_name, 0)
            self.unique_vector_name_counter[vector_name] = unique_id + 1
            label = label + '_uid' + str(unique_id)
        assert constraint_type != ConstraintType.CALLDATA or 'acc' not in kwargs

        if constraint_type == ConstraintType.CALLDATA_ARRAY:
            return z3.Array(label, z3.BitVecSort(bv_size), z3.BitVecSort(8))
        elif constraint_type in [ConstraintType.CALLER, ConstraintType.ORIGIN, ConstraintType.ENTRY_ACCOUNT]:
            return svm_utils.zpad_bv_right(z3.BitVec(label, svm_utils.ADDRESS_LEN), svm_utils.VECTOR_LEN)
        else:
            return z3.BitVec(label, bv_size)

    def get_sym_balance(self, account):
            return z3.BitVec(ConstraintType.BALANCE.value+'_'+str(account.id), 256)

def get_sha_functions(bv_size_in, bv_size_out=256):
    label = 'sha{}'.format(bv_size_in)
    label_inv = label+'_inv'
    sha_func = z3.Function(label, z3.BitVecSort(bv_size_in), z3.BitVecSort(bv_size_out))
    sha_func_inv = z3.Function(label_inv, z3.BitVecSort(bv_size_out), z3.BitVecSort(bv_size_in))
    return sha_func, sha_func_inv
