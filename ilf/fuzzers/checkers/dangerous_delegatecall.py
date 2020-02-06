from .checker import Checker
from ...ethereum import DELEGATECALL


class DangerousDelegatecall(Checker):

    def __init__(self, contract_manager, account_manager):
        super().__init__()
        self.contract_manager = contract_manager
        self.account_manager = account_manager

        self.addresses = []
        for contract in contract_manager.contract_dict.values():
            self.addresses += contract.addresses

        for account in account_manager.accounts:
            self.addresses.append(account.address)


    def check(self, logger):
        logger.addresses = self.addresses
        for i, log in enumerate(logger.logs):
            if log.op == DELEGATECALL:
                args_offset = int(log.stack[-4], 16)
                args_length = int(log.stack[-5], 16)
                value_from_call0 = False
                if args_length != 0 and int.from_bytes(bytes.fromhex(log.memory[2:])[args_offset:args_offset+args_length], byteorder='big') != 0:
                    try:
                        value_from_call0, _ = logger.trace_log_memory(i-1, args_offset, args_offset + args_length)
                    except RecursionError:
                        pass

                try:
                    value_from_call1, _ = logger.trace_log_stack(i-1, -2)
                except RecursionError:
                    value_from_call1 = False

                if value_from_call0 or value_from_call1:
                    return True

        return False
