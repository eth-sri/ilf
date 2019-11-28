from ethereum.opcodes import opcodes
from ilf.symbolic import utils
import re
import sys

regex_PUSH = re.compile('^PUSH(\d+)$')

# Additional mnemonic to catch failed assertions

opcodes[0xfe] = ['ASSERT_FAIL', 0, 0, 0]


# TODO fix it
def get_jumpi_pc_to_func_hash(instruction_list, function_hashes):
    jumpi_pc_to_func_hash = {}
    function_hashes_ints = set([h for h in function_hashes])
    for i in range(0, len(instruction_list)):
        opcode = instruction_list[i]['opcode']
        if re.match(regex_PUSH, opcode):
            arg = int(instruction_list[i]['argument'], 16)
            seen_eq = False
            if arg in function_hashes_ints:
                offset = 1;
                while instruction_list[i+offset]['opcode'] != 'JUMPI':
                    seen_eq = seen_eq or instruction_list[i+offset]['opcode'] == 'EQ'
                    offset += 1
                if offset <= 5 and seen_eq:
                    jumpi_pc_to_func_hash[i+offset] = arg
    return jumpi_pc_to_func_hash

def find_swarmhashes(bytecode):
    swarm_hashes_bytes =  re.findall(b'\\xa1ebzzr.{3}(.{32}).{2}', bytecode, re.DOTALL)
    swarm_hashes_hex = [h.hex() for h in swarm_hashes_bytes]
    return swarm_hashes_hex

def disassemble(bytecode):
    bytecode = re.sub(b'\\xa1ebzzr.{37}', b'\x00' * 43, bytecode)

    instruction_list = []
    addr = 0

    length = len(bytecode)

    if "bzzr" in str(bytecode[-43:]):
        # ignore swarm hash
        length -= 43

    while addr < length:

        instruction = {}

        instruction['address'] = addr

        try:
            if sys.version_info > (3, 0):
                opcode = opcodes[bytecode[addr]]
            else:
                opcode = opcodes[ord(bytecode[addr])]

        except KeyError:

            # invalid opcode
            instruction_list.append({'address': addr, 'opcode': "INVALID"})
            addr += 1
            continue

        instruction['opcode'] = opcode[0]

        match = re.search(regex_PUSH, opcode[0])

        if match:
            argument = bytecode[addr + 1:addr + 1 + int(match.group(1))]
            instruction['argument'] = "0x" + argument.hex()

            addr += int(match.group(1))

        instruction_list.append(instruction)

        addr += 1

    return instruction_list
