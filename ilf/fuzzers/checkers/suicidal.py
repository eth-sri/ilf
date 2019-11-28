from .checker import Checker
from ...ethereum import SELFDESTRUCT


class Suicidal(Checker):

    def __init__(self, contract_manager, account_manager):
        super().__init__()
        self.contract_manager = contract_manager
        self.account_manager = account_manager

        self.attacker_addresses = set()
        for account in self.account_manager:
            self.attacker_addresses.add(int(account.address, 16))

    def check(self, logger):
        for log in logger.logs:
            if log.op == SELFDESTRUCT and int(log.stack[-1], 16) in self.attacker_addresses:
                return True

        return False