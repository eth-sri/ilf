import random


from ..policy_base import PolicyBase
from ..imitation import PolicyImitation
from ..symbolic import PolicySymbolic

class PolicyMix(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager, args):
        super().__init__(execution, contract_manager, account_manager)

        self.policy_fuzz = PolicyImitation(execution, contract_manager, account_manager, args)
        self.policy_fuzz.load_model()
        self.policy_symbolic = PolicySymbolic(execution, contract_manager, account_manager)


    def select_tx(self, obs):
        tx = self.policy_fuzz.select_tx(obs.obs_fuzz)

        if random.random() > 0.9:
            tx_symbolic = self.policy_symbolic.select_tx(obs.obs_symbolic)
            if tx_symbolic is not None:
                tx = tx_symbolic

        return tx