from .checker import Checker


class Locking(Checker):

    def __init__(self, contract_manager, account_manager):
        super().__init__()
        self.contract_manager = contract_manager
        self.account_manager = account_manager

    def check(self, logger):
        can_send_ether = self.contract_manager[logger.tx.contract].can_send_ether
        can_receive_ether = self.contract_manager[logger.tx.contract].can_receive_ether

        return can_receive_ether and not can_send_ether and logger.contract_receive_ether