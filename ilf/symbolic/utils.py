from ilf.ethereum import SolType
from ethereum.utils import big_endian_to_int
from ethereum.abi import encode_abi, encode_int
import logging
import ethereum.abi as abi
from ethereum.abi import method_id
from ethereum.utils import zpad
import binascii
import csv
import json
import os
import pprint
import re
import sys
import yaml
from z3 import simplify

# Keeps track on latest number of addresses created
contract_address_counter = 0
lib_address_counter = 0

ADDRESS_ARG_TAG = '#'

WORD_SIZE = 32
REFERENCE_SEPARATOR = '.'


class AssertionData:
    def __init__(self):
        pass

def process_type(typ):
    t = typ.t
    if t == SolType.IntTy:
        return 'int', typ.size, []
    elif t == SolType.UintTy:
        return 'uint', typ.size, []
    elif t == SolType.BoolTy:
        return 'bool', '', []
    elif t == SolType.StringTy:
        return 'string', '', []
    elif t == SolType.SliceTy:
        base, sub, arr = process_type(typ.elem)
        arr.insert(0, [])
        return base, sub, arr
    elif t == SolType.ArrayTy:
        base, sub, arr = process_type(typ.elem)
        arr.insert(0, [typ.size])
        return base, sub, arr
    elif t == SolType.AddressTy:
        return 'address', '', []
    elif t == SolType.FixedBytesTy:
        return 'bytes', typ.size, []
    elif t == SolType.BytesTy:
        return 'bytes', '', []
    else:
        assert False, 'type {} not supported'.format(t)

def decode_arg(typ, arg, skip_ints=True):
    t = typ.t
    if t == SolType.IntTy:
        return None if skip_ints else arg
    elif t == SolType.UintTy:
        return None if skip_ints else arg
    elif t == SolType.BoolTy:
        return 1 if arg else 0
    elif t == SolType.StringTy:
        value = int.from_bytes(bytes(arg, 'utf-8'), 'big')
        return [len(arg), value]
    elif t == SolType.SliceTy:
        values = [decode_arg(typ.elem, a) for a in arg]
        return values
    elif t == SolType.ArrayTy:
        assert typ.size == len(arg)
        values = [decode_arg(typ.elem, a) for a in arg]
        return values
    elif t == SolType.AddressTy:
        return int(arg[2:], 16)
    elif t == SolType.FixedBytesTy:
        assert typ.size == len(arg)
        value = int.from_bytes(bytes(arg), 'big')
        return value
    elif t == SolType.BytesTy:
        value = int.from_bytes(bytes(arg), 'big')
        return [len(arg), value]
    else:
        assert False, 'type {} not supported'.format(t)

def abi_decode(inputs, calldata_evals, random_eval):
    data = abi_encode(inputs, random_eval, False)
    for i, d in enumerate(data):
        data[i] = calldata_evals.get(i*32, d)
    proctypes = [process_type(i.evm_type) for i in inputs]
    sizes = [abi.get_size(typ) for typ in proctypes]
    outs = [None] * len(inputs)
    start_positions = [None] * len(inputs) + [len(data)]
    for i, _ in enumerate(inputs):
        if sizes[i] is None:
            start_positions[i] = data[i] // 32
            j = i - 1
            while j >= 0 and start_positions[j] is None:
                start_positions[j] = start_positions[i]
                j -= 1
        else:
            outs[i] = data[i]
    j = len(inputs) - 1
    while j >= 0 and start_positions[j] is None:
        start_positions[j] = start_positions[len(inputs)]
        j -= 1
    assert pos <= len(data), "Not enough data for head"
    # Grab the data for tail arguments using the start positions
    # calculated above
    for i, typ in enumerate(inputs):
        if sizes[i] is None:
            offset = start_positions[i]
            next_offset = start_positions[i + 1]
            outs[i] = data[offset:next_offset]
    # Recursively decode them all
    return [dec(proctypes[i], outs[i]) for i in range(len(outs))]

def abi_encode(inputs, args, skip_ints=True):
    headsize = 0
    proctypes = [process_type(i.evm_type) for i in inputs]
    sizes = [abi.get_size(typ) for typ in proctypes]
    for i, arg in enumerate(args):
        if sizes[i] is None:
            headsize += 32
        else:
            headsize += sizes[i]
    myhead, mytail = [], []
    for i, arg in enumerate(args):
        if sizes[i] is None:
            myhead.append(headsize + len(mytail))
            decoded_arg = decode_arg(inputs[i].evm_type, args[i], skip_ints)
            decoded_arg = decoded_arg if isinstance(decoded_arg, list) else [decoded_arg]
            mytail.extend(decoded_arg)
        else:
            decoded_arg = decode_arg(inputs[i].evm_type, args[i], skip_ints)
            decoded_arg = decoded_arg if isinstance(decoded_arg, list) else [decoded_arg]
            myhead.extend(decoded_arg)
    myhead.extend(mytail)
    return myhead


class BColors:
    ENDC = u'\u001b[0m'
    RED = u'\u001b[38;5;1m'
    GREEN = u'\u001b[38;5;2m'
    YELLOW = u'\u001b[38;5;3m'
    BLUE = u'\u001b[38;5;27m'
    BOLD = u'\u001b[1m'
    UNDERLINE = u'\u001b[4m'
    REVERSED = u'\u001b[7m'


def get_dependency_graph(contract_to_constructor):
    dependencies = []
    for contract in contract_to_constructor.keys():
        if contract not in dependencies:
            get_dependencies(contract_to_constructor, contract, dependencies)
    return dependencies


def get_dependencies(contract_to_constructor, contract, dependencies):
    if contract in contract_to_constructor:
        for arg in contract_to_constructor[contract]:
            if type(arg) == str and ADDRESS_ARG_TAG in arg and arg not in dependencies:
                get_dependencies(contract_to_constructor, arg, dependencies)
    dependencies.append(contract)


def format_constraints(constraints):
    res = '['
    for c in constraints:
        res += '\n\t' + str(c).replace('\n', '')
    return res + ']'


def format_model(model, hash_to_func):
    res = ''
    tx_regex = re.compile('_tx(\d+)')
    sorted_decl = sorted(model.decls(), key=lambda d: (tx_regex.findall(str(d)), str(d)))
    for decl in sorted_decl:
        res += str(decl) + ' = ' + str(model[decl]) + '\n'
    res = re.sub(r'([^_])([\d]{4}\d+)', lambda m: m.group(1) + "{0:#0{1}x}".format((int(m.group(2))), 66), str(res))
    # res = re.sub(r'(?:0x)?([\da-f]{8})0+', lambda m: hash_to_func.get(m.group(1).rjust(8, '0'), m.group(1)), res)
    return res


def format_stack(stack):
    stack_string = ''
    for s in stack:
        stack_string += str(simplify(s)).replace('\n', '')
        stack_string += ', '
    stack_string = ' '.join(stack_string.split())
    stack_string = re.sub("([\d]{1}\d+)", lambda m: hex(int(m.group(1))), str(stack_string))
    return stack_string[:-1]


def format_storage(storage):
    storage_string = ' '.join(str(simplify(storage)).split())
    matches = re.findall('\), ([^,\)]+), ([^,\)]+)', storage_string)
    storage_map = {}
    for index, value in matches:
        storage_map[index] = value
    return pprint.pformat(storage_map)


def decode_hex_string(hex_encoded_string):
    if hex_encoded_string.startswith("0x"):
        return bytes.fromhex(hex_encoded_string[2:])
    else:
        return bytes.fromhex(hex_encoded_string)


def encode_calldata(func_name, arg_types, args):
    mid = method_id(func_name, arg_types)
    function_selector = zpad(encode_int(mid), 4)
    args = encode_abi(arg_types, args)
    return '0x' + function_selector.hex() + args.hex()


def encode_sym_abi(types):
    proctypes = [abi.process_type(typ) for typ in types]
    sizes = [abi.get_size(typ) for typ in proctypes]
    headsize = 32 * len(types)
    myhead, mytail = b'', b''
    for i, typ in enumerate(types):
        sym_arg_tag = 'symbolic_argument_' + str(i)
        if sizes[i] is None:
            myhead += abi.enc(abi.INT256, headsize + len(mytail))
            mytail += abi.enc(proctypes[i], sym_arg_tag)
        else:
            myhead += abi.enc(proctypes[i], sym_arg_tag)
    return myhead + mytail


def get_random_address():
    return binascii.b2a_hex(os.urandom(20)).decode('UTF-8')

def get_next_contract_address():
    global contract_address_counter
    contract_address_counter += 1
    return int(('0xC0DE' + hex(contract_address_counter)[2:] + '1' * 40)[0:42], 16)

def get_next_lib_address():
    global lib_address_counter
    lib_address_counter += 1
    return int(('0xABC' + hex(lib_address_counter)[2:] + '1' * 40)[0:42], 16)

def get_caller_indexed_address(index):
    return int(('0xbeef' + hex(index + 1)[2:] + '1' * 40)[0:42], 16)


def write_json_objects_to_file(result, filename):
    print('Writing json objects...')
    with open(filename + '.txt', 'a') as txtfile:
        for res in result:
            txtfile.write(str(res) + ',')


def write_csv_file(result, filename):
    print('Writing csv file...')
    with open(filename + '.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for res in result:
            writer.writerow([res['property'], res['holds'], res['model']])


def parse_yaml_configuration(config_filename):
    data_loaded = {}
    with open(config_filename, 'r') as cf:
        data_loaded = yaml.load(cf)
    return data_loaded


def filter_contract_name(contract_tag):
    if ADDRESS_ARG_TAG in contract_tag:
        return contract_tag.split(ADDRESS_ARG_TAG)[0]
    return contract_tag
