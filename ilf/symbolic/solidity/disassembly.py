from ilf.symbolic.solidity import asm
from ilf.symbolic import utils
import json
import logging
import os


class Disassembly:
    def __init__(self, code, func_hash_to_name=None):
        self.code = code
        self.bytecode = utils.decode_hex_string(code)
        self.instruction_list = asm.disassemble(self.bytecode)
        self.swarm_hashes = asm.find_swarmhashes(self.bytecode)
        self.xrefs = []
        self.func_hash_to_name = func_hash_to_name if func_hash_to_name else {}
        self.func_name_to_hash = {n: h for h, n in self.func_hash_to_name.items()}

        self.func_hash_to_addr = {}
        self.addr_to_func_hash = {}

        self.jumpi_pc_to_func_hash = asm.get_jumpi_pc_to_func_hash(self.instruction_list, list(self.func_name_to_hash.values()))
