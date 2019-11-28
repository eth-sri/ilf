from ilf.symbolic.solidity.disassembly import Disassembly
import persistent
import json


class SolidityContract(persistent.Persistent):
    def __init__(self, name, abi, bytecode, runtime_bytecode, src_map, runtime_src_map, src_code, hashes):
        self.abi = abi
        self.src_map = [runtime_src_map, src_map]
        self.src_code = src_code
        self.name = name
        self.runtime_bytecode = runtime_bytecode
        self.bytecode = bytecode
        self.disassembly = Disassembly(self.runtime_bytecode, hashes)
        self.creation_disassembly = Disassembly(self.bytecode)

    def get_function_arguments_data(self):
        func_to_arg_data = {}
        overloaded_func_names = []
        for entity_data in self.abi:
            if entity_data['type'] == 'function':
                func_name = entity_data['name']
                input_pos = 1
                if func_name not in func_to_arg_data.keys():
                    func_to_arg_data[func_name] = {}
                else:
                    overloaded_count = overloaded_func_names.count(func_name)
                    overloaded_func_names.append(func_name)
                    func_name = func_name + '-' + str(overloaded_count + 1)
                    func_to_arg_data[func_name] = {}

                for input in entity_data['inputs']:
                    func_to_arg_data[func_name][input_pos] = input['type']
                    input_pos += 1

        return func_to_arg_data, overloaded_func_names

    def __repr__(self):
        return "SolidityContract_{}_{}".format(self.name, id(self))

    def __str__(self):
        return str(self.as_dict())

    def as_dict(self):
        return {
            'name': self.name,
            }
