import random
import numpy
import torch
import logging

from ..execution import Execution, Tx
from ..ethereum import Method
from .random import PolicyRandom
from .symbolic import PolicySymbolic
from .sym_plus import PolicySymPlus
from .mix import PolicyMix, ObsMix
from .imitation import PolicyImitation


LOG = logging.getLogger(__name__)


class Environment:

    def __init__(self, limit, seed):
        self.limit = limit
        self.seed = seed


    def fuzz_loop(self, policy, obs):
        obs.init()

        LOG.info(obs.stat)
        LOG.info('initial calls start')
        self.init_txs(policy, obs)
        LOG.info('initial calls end')

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        numpy.random.seed(self.seed)

        for i in range(1, self.limit+1):
            if policy.__class__ in (PolicyRandom, PolicyImitation) and i > self.limit // 2:
                for contract_name in policy.contract_manager.fuzz_contract_names:
                    contract = policy.contract_manager[contract_name]
                    policy.execution.set_balance(contract.addresses[0], 10 ** 29)

            tx = policy.select_tx(obs)
            if tx is None:
                break

            logger = policy.execution.commit_tx(tx)
            old_insn_coverage = obs.stat.get_insn_coverage(tx.contract)
            obs.update(logger, False)
            new_insn_coverage = obs.stat.get_insn_coverage(tx.contract)

            if policy.__class__ in (PolicySymbolic, PolicySymPlus) and new_insn_coverage - old_insn_coverage < 1e-5:
                break

            LOG.info(obs.stat)

            if policy.__class__ not in (PolicySymbolic, PolicySymPlus) and i % 50 == 0:
                policy.reset()
                if policy.__class__ == PolicyImitation:
                    policy.clear_history()
                if policy.__class__ == PolicyMix and policy.policy_fuzz.__class__ == PolicyImitation:
                    policy.policy_fuzz.clear_history()
                if obs.__class__ == ObsMix:
                    obs.reset()


    def init_txs(self, policy, obs):
        policy_random = PolicyRandom(policy.execution, policy.contract_manager, policy.account_manager)

        for name in policy.contract_manager.fuzz_contract_names:
            contract = policy.contract_manager[name]
            if Method.FALLBACK not in contract.abi.methods_by_name:
                tx = Tx(policy_random, contract.name, contract.addresses[0], Method.FALLBACK, bytes(), [], 0, 0, 0, True)
                logger = policy_random.execution.commit_tx(tx)
                obs.update(logger, True)
                LOG.info(obs.stat)

            for method in contract.abi.methods:
                if not contract.is_payable(method.name):
                    tx = policy_random.select_tx_for_method(contract, method, obs)
                    tx.amount = 1
                    logger = policy_random.execution.commit_tx(tx)
                    obs.update(logger, True)
                    LOG.info(obs.stat)