import os
import json


class SolType:

    IntTy = 0
    UintTy = 1
    BoolTy = 2
    StringTy = 3
    SliceTy = 4
    ArrayTy = 5
    AddressTy = 6
    FixedBytesTy = 7
    BytesTy = 8
    HashTy = 9
    FixedPointTy = 10
    FunctionTy = 11

    def __init__(self, *args, **kwargs):
        self.elem = None if kwargs['Elem'] is None else SolType(**kwargs['Elem'])
        self.kind = kwargs['Kind']
        self.size = kwargs['Size']
        self.t = kwargs['T']


    def is_address_type(self):
        if self.t == SolType.AddressTy:
            return True
        elif self.t in (SolType.SliceTy, SolType.ArrayTy, SolType.FixedBytesTy, SolType.BytesTy):
            if self.elem is not None:
                return self.elem.is_address_type()
            else:
                return False


class Argument:

    def __init__(self, *args, **kwargs):
        self.name = kwargs['Name']
        self.evm_type = SolType(**kwargs['Type'])


class Method:

    FALLBACK = 'default'

    def __init__(self, *args, **kwargs):
        self.name = kwargs['Name']
        self.idd = kwargs['ID']
        self.const = kwargs['Const']
        self.inputs = [Argument(**argument) for argument in kwargs['Inputs']] if kwargs['Inputs'] is not None else []
        self.outputs = [Argument(**argument) for argument in kwargs['Outputs']] if kwargs['Outputs'] is not None else []

        self.insns = set()
        self.bow = [0 for _ in range(256)]

        self.entry_block = None
        self.blocks = set()
        self.storage_args = dict()


    def len_args(self):
        return len(self.inputs)


    def num_addrs_in_args(self):
        num = 0
        for inp in self.inputs:
            if inp.evm_type.is_address_type():
                num += 1
        return num


class ABI:

    def __init__(self, *args, **kwargs):
        self.contract = kwargs['contract']
        self.proj_path = kwargs['proj_path']

        self.constructor = Method(**kwargs['Constructor'])
        self.payable = kwargs['payable']
        self.methods = [Method(**method) for method in kwargs['Methods'].values()]

        compiled_json_path = os.path.join(self.proj_path, 'build', 'contracts', '{}.json'.format(self.contract.name))
        with open(compiled_json_path) as compiled_json_f:
            compiled_json = json.load(compiled_json_f)
            abi_json = compiled_json['abi']
            for item in abi_json:
                if item['type'] == 'fallback':
                    self.methods.append(Method(Name=Method.FALLBACK, ID=None, Const=False, Inputs=None, Outputs=None))
                    if item['payable']:
                        self.payable[Method.FALLBACK] = True
                    else:
                        self.payable[Method.FALLBACK] = False

        self.methods_by_name = dict()
        self.methods_by_idd = dict()
        for method in self.methods:
            self.methods_by_name[method.name] = method
            self.methods_by_idd[method.idd] = method