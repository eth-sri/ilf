from enum import Enum
import z3
import ethereum


class CalldataType(Enum):
    DEFINED = 1
    UNDEFINED = 2


"""
WARINING

Environment sould stay immutable.
"""


class Environment:
    def __init__(
            self,
            active_address,
            sender,
            calldata,
            gasprice,
            callvalue,
            origin,
            calldata_type=CalldataType.UNDEFINED,
            disassembly=None,
            runtime_bytecode_bytes=None,
            timestamp=None
    ):
        self.active_address = active_address
        self.sender = sender
        self.calldata = calldata
        self.gasprice = gasprice
        self.callvalue = callvalue
        self.origin = origin
        self.calldata_type = calldata_type
        self.disassembly = disassembly
        self.runtime_bytecode_bytes = runtime_bytecode_bytes
        self.timestamp = timestamp

    def __str__(self):
        return str({'active_address': self.active_address,
                    'sender': self.sender,
                    'calldata': self.calldata,
                    'gasprice': self.gasprice,
                    'callvalue': self.callvalue,
                    'origin': self.origin,
                    'calldata_type': self.calldata_type,
                    'timestamp': self.timestamp})

    def extract_func_name_hash(self, jumpi_pc):
        func_name = None
        func_hash = None
        jumpi_pc_to_func_hash = self.disassembly.jumpi_pc_to_func_hash
        if jumpi_pc in jumpi_pc_to_func_hash:
            func_hash = jumpi_pc_to_func_hash[jumpi_pc]
            func_hash_to_name = self.disassembly.func_hash_to_name
            if func_hash in func_hash_to_name:
                func_name = func_hash_to_name[func_hash]
            else:
                func_name = "_function_" + hex(func_hash)[2:]
        return func_name, func_hash
