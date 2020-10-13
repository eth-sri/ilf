from .checker import Checker
from ...ethereum import *


class Reentrancy(Checker):

    def __init__(self, contract_manager, account_manager):
        super().__init__()
        self.contract_manager = contract_manager
        self.account_manager = account_manager

    def check(self, logger):
        has_transfer = False
        change_state = False
        pc1, pc2 = -1, -1
        
        for log in logger.logs:
            if log.op == CALL and int(log.stack[-3], 16) > 0 and int(log.stack[-1], 16) > 0 :  # CALL: [gas  addr  value  argsOffset  argsLength  retOffset  retLength]
                has_transfer = True
                pc1 = log.pc
            if log.op == SSTORE :
                change_state = True
                pc2 = log.pc
        
        pc_follow = ((pc1 != -1) and (pc2 != -1) and (pc1 < pc2))

        return has_transfer and change_state and pc_follow
