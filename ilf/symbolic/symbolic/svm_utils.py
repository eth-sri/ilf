from collections import defaultdict
import ethereum
import itertools
from ilf.symbolic.solidity.soliditycontract import SolidityContract
from ilf.symbolic.solidity import solidity_utils
from ilf.symbolic.symbolic import constraints
from ilf.symbolic.symbolic import svm_utils
from ilf.symbolic import utils
import z3
import z3.z3util
import logging
import json
import os
import re
import time
import datetime
import multiprocessing as mp
import sys

VECTOR_LEN = 256
ADDRESS_LEN = 160
TT256 = 2 ** VECTOR_LEN
TT256M1 = 2 ** VECTOR_LEN - 1
TT255 = 2 ** (VECTOR_LEN - 1)
# 200B ether max. 10**18 - wei convertion
ETHER_LIMIT = 10 ** 29


SIMPLE_STORAGE_SYMBOLIC_TAG = 'simple_sym'
MAPPING_STORAGE_PREFIX = 'mapping_'

def get_zero_array(index_size=VECTOR_LEN, value_size=VECTOR_LEN):
    return z3.K(z3.BitVecSort(index_size), z3.BitVecVal(0, value_size))

EMPTY_ARRAY = get_zero_array()



INIT_FUNC_NAME = 'init()'
FALLBACK_FUNC_NAME = 'fallback()'


def construct_trace_access_sets(wstates):
    raise Exception('Broken')
    trace_to_read_set = defaultdict(set)
    trace_to_write_set = defaultdict(set)
    for wstate in wstates:
        for address_index in wstate.sstore_address_index:
            trace_to_read_set[wstate.trace].add(address_index)
        for address_index in wstate.sload_address_index:
            trace_to_write_set[wstate.trace].add(address_index)
    return trace_to_read_set, trace_to_write_set


def extract_trace_to_independent_traces(wstates):
    trace_to_read_set, trace_to_write_set = construct_trace_access_sets(wstates)
    all_traces = trace_to_write_set.keys() | trace_to_read_set.keys()
    trace_to_independent_traces = defaultdict(set)
    trace_pairs = itertools.combinations(all_traces, 2)
    for trace, other_trace in trace_pairs:
            if (sets_independent(trace_to_write_set[trace], trace_to_write_set[other_trace]) and
                    sets_independent(trace_to_read_set[trace], trace_to_write_set[other_trace]) and
                    sets_independent(trace_to_write_set[trace], trace_to_read_set[other_trace])):
                trace_to_independent_traces[trace].add(other_trace)
                trace_to_independent_traces[other_trace].add(trace)
    return trace_to_independent_traces


# Trace A depends on B if: A reads from indexes B writes to
def extract_trace_to_dependent_traces(wstates):
    trace_to_read_set, trace_to_write_set = construct_trace_access_sets(wstates)
    all_traces = trace_to_write_set.keys() | trace_to_read_set.keys()
    trace_to_dependent_traces = defaultdict(set)
    trace_pairs = itertools.permutations(all_traces, 2)
    for a_trace, b_trace in trace_pairs:
        reads_a_in_writes_b = trace_to_read_set[a_trace] & trace_to_write_set[b_trace]
        if len(reads_a_in_writes_b):
            trace_to_dependent_traces[a_trace].add(b_trace)
    return trace_to_dependent_traces


def extract_storage_indexes(wstates):
    raise Exception('Broken')
    address_index_set = set()
    for wstate in wstates:
        address_index_set.update(wstate.sstore_address_index)
        address_index_set.update(wstate.sload_address_index)
    return address_index_set

def draw_wstate_tree(svm):
    import matplotlib.pyplot as plt
    import networkx as nx
    from networkx.drawing.nx_agraph import write_dot, graphviz_layout

    G = nx.DiGraph()
    pending_list = [svm.root_wstate]
    while len(pending_list):
        root = pending_list.pop()
        for trace, children in root.trace_to_children.items():
            for c in children:
                G.add_edge(repr(root), repr(c), label=trace)
                pending_list.append(c)
    # pos = nx.spring_layout(G)
    pos = graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.show()


def make_trace(account, func_name):
    trace = '{}.{}'.format(account.contract.name, func_name)
    return trace

def expand_interest_traces(interest_traces, trace_to_dependent_traces):
    all_interest_traces = set()
    pending_traces = list(interest_traces)
    while(len(pending_traces)):
        new_trace = pending_traces.pop()
        if new_trace not in all_interest_traces:
            all_interest_traces.add(new_trace)
            for dt in trace_to_dependent_traces[new_trace]:
                pending_traces.append(dt)
    return all_interest_traces


def sets_independent(a, b):
    """ Returns True if storage access sets are independent"""
    a_has_sym_tag = SIMPLE_STORAGE_SYMBOLIC_TAG in a
    b_has_sym_tag = SIMPLE_STORAGE_SYMBOLIC_TAG in b
    if a_has_sym_tag and b_has_sym_tag:
        return False
    elif a_has_sym_tag:
        return len(b) == 0
    elif b_has_sym_tag:
        return len(a) == 0
    else:
        return len(a & b) == 0

def convert_to_signed(i):
    return i if i < TT255 else i - TT256

def resolve_address(wstate, address_bv, frame_predicates=[]):
    if is_bv_concrete(address_bv):
        return get_concrete_int(address_bv)
    frame_constraints = [wstate.predicate_to_constraints[fpred] for fpred in frame_predicates]
    solver = z3.Solver()
    solver.set('timeout', 2000)
    solver.add(frame_constraints)
    res = solver.check()
    assert res != z3.unsat
    model = solver.model()
    address = model.evaluate(address_bv)
    if not is_bv_concrete(address):
        return None
    address = address.as_long()
    return  address if address in wstate.address_to_account else None


def resolve_address_old(address):
    try:
        string_address = str(z3.simplify(address))
        if constraints.ConstraintType.CREATED_CONTRACT.value in string_address:
            matches = re.findall(r'(0x[a-f0-9]+)', string_address)
            assert len(matches)==1, 'Multiple matches, cannot resolve address'
            return matches[0]
        else:
            return hex(int(string_address))
    except Exception as e:
        # logging.warning('Address %s could not be resolved properly.', address)
        return None


def get_abi_types(abi):
    abi_constructor = None
    for abi_element in abi:
        if abi_element['type'] == 'constructor':
            abi_constructor = abi_element  # TODO: only one such type? Then have a break?
    arg_types = []
    if abi_constructor:
        for arg in abi_constructor['inputs']:
            arg_types.append(arg['type'])
    return arg_types


def get_instruction_index(instruction_list, address):
    # TODO add a dict to instruction list
    index = 0
    for instr in instruction_list:
        if instr['address'] == address:
            return index
        index += 1
    return None


def get_trace_line(instr, state):
    stack = str(state.stack[::-1])
    stack = re.sub("\n", "", stack)
    return str(instr['address']) + " " + instr['opcode'] + "\tSTACK: " + stack

def convert_to_bitvec(val):
    if isinstance(val, z3.BoolRef):
        return z3.If(val, z3.BitVecVal(1, 256), z3.BitVecVal(0, 256))
    elif isinstance(val, bool):
        return z3.BitVecVal(1, 256) if val else z3.BitVecVal(0, 256)
    elif isinstance(val, int):
        return z3.BitVecVal(val, 256)
    else:
        return z3.simplify(val)

def get_memory_data(memory, index, length):
    assert type(index) == int and type(length) == int, 'Wrong types for get_memory_data'
    value_bytes = []
    for i in range(length):
        value_bytes.append(z3.Select(memory, index+i))
    value = z3.Concat(value_bytes)
    assert value.size() == (length * 8)
    return value

def is_bv_concrete(bv):
    if isinstance(bv, int):
        return True
    try:
        hex(z3.simplify(bv).as_long())
        return True
    except AttributeError as e:
        return False
    except Exception:
        raise Exception("pdb")


def extract_storage_slicing_index(bv):
    if is_bv_concrete(bv) and get_concrete_int(bv) < 100:
        return [get_concrete_int(bv)]
    else:
        return extract_sha_slicing_index(bv)

def extract_sha_slicing_index(bv):
    if bv.decl().name() == 'sha512':
        sha_args = bv.arg(0)
        _, mapping_id = split_bv(sha_args, 256)
        if is_bv_concrete(mapping_id):
            return [get_concrete_int(mapping_id)]
    elif bv.decl().name() == 'sha256' and is_bv_concrete(bv.arg(0)):
        return [get_concrete_int(bv.arg(0))]
    found_shas = []
    for c in bv.children():
        c_res = extract_sha_slicing_index(c)
        found_shas.extend(c_res)
    return found_shas

def extract_concrete_prefix_index(bv):
    try:
        get_concrete_int(bv)
        return -1
    except AttributeError as e:
        pass
    hi = bv.size() - 1
    lo = 0
    while hi > lo:
        mid = lo + (hi - lo) // 2
        hi_bv = z3.Extract(hi, mid, bv)
        lo_bv = z3.Extract(mid-1, lo, bv)
        hi_concrete = is_bv_concrete(hi_bv)
        lo_concrete = is_bv_concrete(lo_bv)
        if hi_concrete and not lo_concrete:
            hi = mid - 1
        elif not hi_concrete and not lo_concrete:
            lo = mid
        else:
            return -1
    return mid

def split_bv(bv, index):
    length = bv.size()
    assert 0 < index and index < length
    hi_bv = z3.Extract(length - 1, index, bv)
    lo_bv = z3.Extract(index - 1, 0, bv)
    return hi_bv, lo_bv


def get_concrete_int(item):
    if type(item) == int:
        return item
    if type(item) == z3.BitVecNumRef:
        return item.as_long()
    return z3.simplify(item).as_long()


def get_concrete_int_from_bytes(_bytes, start_index):
    return int.from_bytes(_bytes[start_index:start_index + 32], byteorder='big')


def convert_concrete_int_to_bytes(val):
    if type(val) == int:
        return val.to_bytes(32, byteorder='big')
    return (z3.simplify(val).as_long()).to_bytes(32, byteorder='big')


def find_library_names(bytecode):
    lib_pattern = re.compile('(.*\.sol:)([a-zA-Z0-9]+)(_+)')
    return set([lib_name for (_, lib_name, _) in re.findall(lib_pattern, bytecode)])


def link_libraries(filename, library_to_address, bytecode):
    replaced_libraries_code = bytecode
    for lib_name, address in library_to_address.items():
        pattern = re.compile('_+' + os.path.basename(filename) + ':' + lib_name + '_+')
        replaced_libraries_code = re.sub(pattern, hex(address)[2:], replaced_libraries_code)
    return replaced_libraries_code


def is_sha_bv(bv):
    sha_decl_re = re.compile('(^sha\d*$)')
    return len(re.findall(sha_decl_re, bv.decl().name())) == 1

def solve_gstate(gstate):
    return solve_wstate(gstate.wstate)

def solve_wstate(wstate):
    solver = z3.Solver()
    solver.add(wstate.constraints)
    res = solver.check()
    if res == z3.unknown: logging.info('gstate check timeout')
    return solver if res == z3.sat else None

def get_library_names(combined_json_data):
    lib_names = []
    filename = combined_json_data['sourceList'][0]
    assert len(combined_json_data['sourceList']) == 1, 'Not a flat file!'
    pattern = re.compile('(' + filename + ':)([a-zA-Z0-9]+)(_+)')
    hash_pattern = re.compile('__\$([a-f0-9]{34})\$__')
    hash_to_contract_name = { ethereum.utils.sha3(c).hex()[:34]: c for c in combined_json_data['contracts'].keys() }
    for contract_name, contract_data in combined_json_data['contracts'].items():
        lib_names.extend([lib_name for (_, lib_name, _) in re.findall(pattern, contract_data['bin-runtime'])])
        hash_matches = re.findall(hash_pattern, contract_data['bin-runtime'])
        for hash_match in hash_matches:
            lib_full_name = hash_to_contract_name[hash_match]
            lib_names.append(lib_full_name.split(':')[-1])
    return set(lib_names)

def replace_libraries_with_address(filename, library_to_address, code):
    replaced_libraries_code = code
    for lib_name, address in library_to_address.items():
        pattern = re.compile('_+' + os.path.basename(filename) + ':' + lib_name + '_+')
        replaced_libraries_code = re.sub(pattern, hex(address)[2:], replaced_libraries_code)
        lib_hash = ethereum.utils.sha3(f'{filename}:{lib_name}').hex()[:34]
        hash_pattern = re.compile(f'__\${lib_hash}\$__')
        replaced_libraries_code = re.sub(hash_pattern, hex(address)[2:], replaced_libraries_code)
    return replaced_libraries_code


# Gathers contracts data and updates it with library addresses
def generate_contract_objects(contract_to_build_data, hash_to_func_name):
    contract_name_to_contract = {}
    lib_name_to_address = {}
    # filename = combined_json_data['sourceList'][0]
    # assert len(combined_json_data['sourceList']) == 1, 'Not a flat file!'
    utils.get_next_lib_address()
    for contract_name, contract_json_data in contract_to_build_data.items():
        src_code = contract_json_data['source']
        runtime_bytecode = contract_json_data['deployedBytecode']
        bytecode = contract_json_data['bytecode']
        for lib_name in find_library_names(bytecode) | find_library_names(runtime_bytecode):
            lib_name_to_address.setdefault(lib_name, utils.get_next_lib_address())
        # runtime_bytecode = link_libraries(filename, lib_name_to_address, runtime_bytecode)
        # bytecode = link_libraries(filename, lib_name_to_address, bytecode)
        abi = contract_json_data['abi']
        runtime_src_map = contract_json_data['deployedSourceMap']
        src_map = contract_json_data['sourceMap']
        contract = SolidityContract(contract_name, abi, bytecode,
                                    runtime_bytecode, src_map,
                                    runtime_src_map, src_code, hash_to_func_name)
        contract_name_to_contract[contract_name] = contract
    return contract_name_to_contract, lib_name_to_address

def resolve_swarmhashes(contract_name_to_contract):
    unresolved_contracts = list(contract_name_to_contract.values())
    contract_to_swarmhashes = defaultdict(set)
    for c in unresolved_contracts:
        contract_to_swarmhashes[c].update(c.creation_disassembly.swarm_hashes)
    swarm_hash_to_contract = {}
    while len(unresolved_contracts):
        for c in unresolved_contracts:
            for sh in swarm_hash_to_contract.keys():
                contract_to_swarmhashes[c].discard(sh)
            if len(contract_to_swarmhashes[c]) == 0:
                unresolved_contracts.remove(c)
            elif len(contract_to_swarmhashes[c]) == 1:
                swarm_hash_to_contract[contract_to_swarmhashes[c].pop()] = c
                unresolved_contracts.remove(c)
    return swarm_hash_to_contract


def find_all_paths(root_wstate):
    all_paths = []

    def find_all_paths_helper(root_wstate, curr_path):
        curr_path.append(root_wstate)
        if len(root_wstate.trace_to_children) == 0:
            all_paths.append(curr_path)
        else:
            for children in root_wstate.trace_to_children.values():
                for child in children:
                    find_all_paths_helper(child, curr_path.copy())

    find_all_paths_helper(root_wstate, [])
    return all_paths

def zpad_bv_right(bv, target_len):
    bv_len = bv.size()
    if bv_len == target_len:
        return bv
    elif bv_len < target_len:
        return z3.Concat(z3.BitVecVal(0, target_len-bv_len), bv)
    else:
        raise ValueError('Target length is less then vector size!')

def zpad_bv_left(bv, target_len):
    bv_len = bv.size()
    if bv_len == target_len:
        return bv
    elif bv_len < target_len:
        return z3.Concat(bv, z3.BitVecVal(0, target_len-bv_len))
    else:
        raise ValueError('Target length is less then vector size!')


# Checks if wstate_a subsumes wstate_b
def check_substitution(wstate_a, wstate_b):
    logging.info('Checking if %s subsumes %s', wstate_a.trace, wstate_b.trace)
    if len(set(wstate_b.address_to_account)-set(wstate_a.address_to_account)):
        logging.debug('check_substitution: Different gstate number')
        return False
    constraints_a = wstate_a.constraints
    constraints_b = wstate_b.constraints
    sum_vars = []
    focus_a = []
    focus_b = []
    mapping_ids = set()
    x_index = 0
    y_index = 1
    x_var = z3.BitVec('X_var', 256)
    # y_var = z3.BitVec('Y_var', 256)
    focus_constraints_a = []
    focus_constraints_b = []
    for address in wstate_b.address_to_account:
        gstate_a = wstate_a.address_to_account[address]
        gstate_b = wstate_b.address_to_account[address]
        focus_constraints_a.append(x_var == z3.simplify(gstate_a.storage.load(x_index)))
        # focus_constraints_a.append(y_var == gstate_a.storage.load(y_index))
        focus_constraints_b.append(x_var == z3.simplify(gstate_b.storage.load(x_index)))
        # focus_constraints_b.append(y_var == gstate_b.storage.load(y_index))
    focus_constraints_a = z3.And(focus_constraints_a)
    focus_constraints_b = z3.And(focus_constraints_b)
    constraints_a = z3.And(constraints_a)
    constraints_b = z3.And(constraints_b)
    vars_a = get_wstate_z3vars(wstate_a)
    vars_b = get_wstate_z3vars(wstate_b)
    vars_a = set([v.n for v in vars_a])
    vars_b = set([v.n for v in vars_b])
    substitution_dict_a = substitute_bvset(vars_a, 'A')
    substitution_dict_b = substitute_bvset(vars_b, 'B')

    # hash_constraints_a = z3.substitute(hash_constraints, [(bv, bv_s) for bv, bv_s in substitution_dict_a.items()])
    # hash_constraints_b = z3.substitute(hash_constraints, [(bv, bv_s) for bv, bv_s in substitution_dict_b.items()])
    sub_pairs_a = [(bv, bv_s) for bv, bv_s in substitution_dict_a.items()]
    sub_pairs_b = [(bv, bv_s) for bv, bv_s in substitution_dict_b.items()]
    focus_a = z3.substitute(focus_constraints_a, sub_pairs_a)
    focus_b = z3.substitute(focus_constraints_b, sub_pairs_b)

    constraints_a_s = z3.substitute(constraints_a, sub_pairs_a)
    constraints_b_s = z3.substitute(constraints_b, sub_pairs_b)
    
    B = z3.And(focus_b, constraints_b_s)
    nA1 = z3.Not(focus_a)
    nA2 = z3.Not(constraints_a_s)

    abstraction_not_implies = z3.Or(z3.And(B, nA1), z3.And(B, nA2))

    bv_solver = z3.Solver()
    bv_solver.set('timeout', 10000)

    a_vars = list(substitution_dict_a.values())
    b_vars = list(substitution_dict_b.values())
    q_vars = [x_var]
    q_vars.extend(b_vars)
    bv_solver.add(z3.simplify(z3.ForAll(q_vars, z3.Exists(a_vars, z3.Implies(z3.And(focus_b), z3.And(focus_a))))))
    res = bv_solver.check()
    print(res)
    assert res == z3.sat or res == z3.unsat
    return res == z3.sat
    # if s.check() != z3.sat and s.check()!= z3.unsat:



def substitute_bvset(bvset, prefix):
    index = 0
    substitution_dict = {}
    for bv in bvset:
        if type(bv) == z3.BitVecRef:
            substitution_dict[bv] = z3.BitVec('{}_{}_{}'.format(bv, prefix, index), bv.size())
        elif type(bv) == z3.ArrayRef:
            substitution_dict[bv] = z3.Array('{}_{}_{}'.format(bv, prefix, index), bv.sort().domain(), bv.sort().range())
        index += 1

    return substitution_dict


# Wrapper for allowing Z3 ASTs to be stored into Python Hashtables.
class AstRefKey:
    def __init__(self, n):
        self.n = n
    def __hash__(self):
        return self.n.hash()
    def __eq__(self, other):
        return self.n.eq(other.n)
    def __repr__(self):
        return str(self.n)

def askey(n):
    assert isinstance(n, z3.AstRef)
    return AstRefKey(n)

def get_z3vars(f):
    r = set()
    def collect(f):
      if z3.is_const(f):
          if f.decl().kind() == z3.Z3_OP_UNINTERPRETED and not askey(f) in r:
              r.add(askey(f))
      else:
          for c in f.children():
              collect(c)
    collect(f)
    return r

def get_wstate_z3vars(wstate):
    r = get_z3vars(z3.simplify(z3.And(wstate.constraints)))
    for address, gstate in wstate.address_to_account.items():
        r.update(get_z3vars(z3.simplify(gstate.storage.storage)))
    return r

def extract_index_features(index):
    current_index = index
    features = []
    if current_index.decl().name() == 'bvadd':
       current_index = current_index.arg(0)
    if current_index.decl().name().startswith('sha'):
        current_index = current_index.arg(0)
        features.append(current_index.size())
    else:
        return features
    if current_index.decl().name() == 'concat':
        args = [current_index.arg(i) for i in range(current_index.num_args())]
        if svm_utils.is_bv_concrete(args[-1]):
            features.append(args[-1])
            current_index = z3.Concat(args[0:-1]) if len(args) > 2 else args[0]
        else:
            return features
    elif is_bv_concrete(current_index):
        if current_index.size() > 256:
            prefix, suffix  = split_bv(current_index, 256)
            suffix = z3.simplify(suffix)
            features.append(suffix)
            current_index = z3.simplify(prefix)
        elif current_index.size() == 256:
            features.append(current_index)
            return features
        else:
            features.append(current_index)
            return features
    else:
        return features
    if current_index.decl().name().startswith('sha'):
        print(current_index)
        current_index = current_index.arg(0)
        features.append(current_index.size())
    else:
        return features
    return features

def split_bv_by_words(bv):
    assert bv.size() % 256 == 0
    bv_bytes = []
    for i in range(bv.size(), 0, -256):
        bv_bytes.append(z3.simplify(z3.Extract(i-1, i-256, bv)))
    return bv_bytes

def split_bv_into_bytes(bv):
    if type(bv) == int:
        bv = z3.BitVecVal(bv, 256)
    assert bv.size() % 8 == 0
    is_conc = False
    if is_bv_concrete(bv):
        is_conc = True
        length = bv.size() // 8
        concrete_data = get_concrete_int(bv)
        data_bytes = ethereum.utils.zpad(ethereum.utils.int_to_bytes(concrete_data), length)
        bv_bytes = []
        for data_byte in data_bytes:
            bv_bytes.append(z3.BitVecVal(data_byte, 8))
        bv_bytes_a  = bv_bytes
    bv_bytes = []
    for i in range(bv.size(), 0, -8):
        bv_bytes.append(z3.simplify(z3.Extract(i-1, i-8, bv)))
    if is_conc:
        assert bv_bytes == bv_bytes_a
    assert len(bv_bytes) == bv.size() // 8
    return bv_bytes

def symbolic_keccak(svm, data):
    sha_constraints = []
    sha_func, sha_func_inv = constraints.get_sha_functions(data.size())
    hash_vector = sha_func(data)

    sha_constraints.append(sha_func_inv(sha_func(data)) == data)
    hash_vector_features = extract_index_features(hash_vector)
    data_concrete = svm_utils.is_bv_concrete(data)
    if data_concrete:
        concrete_data = svm_utils.get_concrete_int(data)
        data_bytes = ethereum.utils.zpad(ethereum.utils.int_to_bytes(concrete_data), data.size()//8)
        hash_value = int.from_bytes(ethereum.utils.sha3_256(data_bytes), 'big')



    SIZE_PER_SHA_LEN = 2**100

    limit_left = 1024 + SIZE_PER_SHA_LEN * data.size()
    limit_right = limit_left + SIZE_PER_SHA_LEN

    if not data_concrete:
        sha_constraints.append(z3.ULT(limit_left, hash_vector))
        sha_constraints.append(z3.ULT(hash_vector, limit_right))
        # last 4 bits are 0 => hashes are 16 words between each other
        sha_constraints.append(z3.Extract(3, 0, hash_vector) == 0)
    elif data_concrete:
        storage_range = limit_right - limit_left
        scaled_hash_value = limit_left + int((hash_value/svm_utils.TT256M1)*storage_range)
        scaled_hash_value = scaled_hash_value // 16 * 16
        sha_constraints.append(hash_vector == z3.BitVecVal(scaled_hash_value, VECTOR_LEN))
    # elif storage_node == svm.storage_root and data_concrete:
        # hash_value = hash_value // 16 * 16
        # sha_constraints.append(hash_vector == z3.BitVecVal(hash_value, VECTOR_LEN))
    return sha_constraints, hash_vector

def is_bv_pow2(bv):
    if is_bv_concrete(bv):
        a = svm_utils.get_concrete_int(bv)
        return a != 0 and ((a-1) & a) == 0
    else:
        return False

def check_wstate_reachable(wstate, timeout=None):
    s = z3.Solver()
    if timeout is not None:
        s.set('timeout', timeout)
    s.add(wstate.constraints)
    res = s.check()
    if res != z3.unsat:
        return True
    else:
        logging.debug(f'deleted wstate {wstate.trace}; len constraints:{len(wstate.constraints)}')
        return False

def decode_abi(types, gen, sym_bv_generator):
    proctypes = [ethereum.abi.process_type(typ) for typ in types]
    sizes = [ethereum.abi.get_size(typ) for typ in proctypes]
    arg_data = [None] * len(types)
    lengths = [None] * len(types)
    pos = 4
    for i, typ in enumerate(types):
        if sizes[i] is None:
            start_offset = sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA, gen, index=pos)
            arg_data[i] = sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA, gen, index=start_offset)
        else:
            arg_data[i] = sym_bv_generator.get_sym_bitvec(constraints.ConstraintType.CALLDATA, gen, index=pos)
        pos += 32
    return arg_data
