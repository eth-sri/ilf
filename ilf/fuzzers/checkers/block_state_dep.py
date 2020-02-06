from .checker import Checker
from ...ethereum import *


class BlockStateDep(Checker):

    def __init__(self):
        super().__init__()


    def check(self, logger):
        block_state_op_idx = -1

        for i, log in enumerate(logger.logs):
            if log.op in (COINBASE, TIMESTAMP, NUMBER, DIFFICULTY, GASLIMIT):
                block_state_op_idx = i
                break

        if block_state_op_idx == -1:
            return False

        last_send_ether_idx = -1
        for i in range(block_state_op_idx, len(logger.logs)):
            log = logger.logs[i]
            if log.op == CREATE:
                value = int(log.stack[-1], 16)
                if value > 0:
                    last_send_ether_idx = max(last_send_ether_idx, i)
            elif log.op in (CALL, CALLCODE):
                value = int(log.stack[-3], 16)
                if value > 0:
                    last_send_ether_idx = max(last_send_ether_idx, i)

        if last_send_ether_idx == -1:
            return False

        for i in range(block_state_op_idx, last_send_ether_idx + 1):
            log = logger.logs[i]
            if log.op == CREATE:
                value = int(log.stack[-1], 16)
                if value == 0:
                    continue

                try:
                    _, value_from_block = logger.trace_log_stack(i-1, -1)
                    if value_from_block:
                        return True
                except RecursionError:
                    pass
            elif log.op in (CALL, CALLCODE):
                value = int(log.stack[-3], 16)
                if value == 0:
                    continue

                try:
                    _, value_from_block = logger.trace_log_stack(i-1, -3)
                    if value_from_block:
                        return True
                except RecursionError:
                    continue
            elif log.op == JUMPI:
                try:
                    _, value_from_block = logger.trace_log_stack(i-1, -2)
                    if value_from_block:
                        return True
                except RecursionError:
                    continue

        return False