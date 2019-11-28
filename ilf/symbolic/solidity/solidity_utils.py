from subprocess import Popen, PIPE
from ilf.symbolic.exceptions import CompilerError
import json
import logging
import os
import sys

COMBINED_JSON_FILENAME = 'combined.json'
COMPACT_AST_SUFFIX = '_json.ast'


def solc_exists(version):
    solc_binary = os.path.join(os.environ['HOME'], '.py-solc/solc-v' + version, 'bin/solc')
    if os.path.exists(solc_binary):
        return True
    else:
        return False



def compile_sol_file(filename, solc_binary):
    solc_binary = solc_binary if solc_binary is not None else 'solc'
    file_dirname = os.path.dirname(filename) or '.'
    file_basename = os.path.basename(filename)
    cmd = [solc_binary,
           # '--optimize', '--optimize-runs', '1000',
           '--evm-version', 'byzantium',
           '--combined-json', 'asm,ast,bin,bin-runtime,srcmap-runtime,srcmap,hashes,abi',
           '--allow-paths', '.']
    cmd.append(file_basename)

    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=file_dirname)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise CompilerError('Solc experienced a fatal error (code %d).\n\n%s' % (p.returncode, stderr.decode('UTF-8')))
    except FileNotFoundError:
        raise CompilerError('Solc not found.')

    out = stdout.decode('UTF-8')
    assert len(out), 'Compilation failed.'
    combined_json_data = json.loads(out)
    return combined_json_data

def offset_to_line(source_code, bytecode_offset, source_mapping):
    srcmap_runtime_mappings = source_mapping[0].split(';')
    srcmap_mappings = source_mapping[1].split(';')
    mappings = None
    if bytecode_offset < 0 or len(srcmap_mappings) <= bytecode_offset:
        if bytecode_offset < 0 or len(srcmap_runtime_mappings) <= bytecode_offset:
            logging.debug('Bytecode offset is wrong!')
            return 0
        else:
            mappings = srcmap_runtime_mappings
    else:
        mappings = srcmap_mappings

    src_offset = -1
    while True:
        src_offset = mappings[bytecode_offset].split(':')[0]
        bytecode_offset -= 1
        if not ((src_offset == '' or int(src_offset) < 0) and bytecode_offset >= 0):
            break
    if src_offset != '' and int(src_offset) >= 0:
        source_line = get_source_line_from_offset(source_code, int(src_offset))
        return source_line


def get_source_line_from_offset(source_code, src_offset):
    linebreaks = 0
    for line, content in enumerate(source_code):
        if line > src_offset:
            break
        if content == '\n':
            linebreaks += 1
    return linebreaks
