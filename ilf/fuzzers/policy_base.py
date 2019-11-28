import abc
import random


class PolicyBase:

    def __init__(self, execution, contract_manager, account_manager):
        self.execution = execution
        self.contract_manager = contract_manager
        self.account_manager = account_manager

        self.addresses = []
        for contract in contract_manager.contract_dict.values():
            self.addresses += contract.addresses

        for account in account_manager.accounts:
            self.addresses.append(account.address)


    @abc.abstractmethod
    def select_tx(self, obs):
        raise NotImplementedError


    def reset(self):
        self.execution.jump_state(0)


    def jump_state(self, idd):
        self.execution.jump_state(idd)


    def _select_timestamp(self, obs):
        return random.randint(1438214400, 1753833600)