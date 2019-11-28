import json
import ctypes

from ..ethereum import ContractManager, AccountManager
from .logger import Logger
from ..ethereum import Method


class Execution:

    def __init__(self, path):
        self.path = path
        self.lib = ctypes.cdll.LoadLibrary(path)

        self.lib.SetBackend.argtypes = [ctypes.c_char_p]
        self.lib.SetBackend.restype = ctypes.c_char_p

        self.lib.GetContracts.argtypes = []
        self.lib.GetContracts.restype = ctypes.c_char_p

        self.lib.GetAccounts.argtypes = []
        self.lib.GetAccounts.restype = ctypes.c_char_p

        self.lib.CommitTx.argtypes = [ctypes.c_char_p]
        self.lib.CommitTx.restype = ctypes.c_char_p

        self.lib.JumpState.argtypes = [ctypes.c_int]
        self.lib.JumpState.restype = None

        self.lib.SetBalance.argtypes = [ctypes.c_char_p]
        self.lib.SetBalance.restype = None


    def set_backend(self, proj_path):
        proj_path = proj_path.encode('ascii')
        bs = self.lib.SetBackend(proj_path)
        j = json.loads(bs.decode())
        loggers = [Logger(**l) for l in j]
        return loggers


    def get_contracts(self):
        bs = self.lib.GetContracts()
        j = json.loads(bs.decode())
        return ContractManager(**j)


    def get_accounts(self):
        bs = self.lib.GetAccounts()
        j = json.loads(bs.decode())
        manager = AccountManager(**j)
        return manager


    def commit_tx(self, tx):
        if tx.method == Method.FALLBACK:
            tx.method = ''
        tx = tx.to_execution_str().encode('ascii')
        bs = self.lib.CommitTx(tx)
        j = json.loads(bs.decode())
        logger = Logger(**j)
        if logger.tx.method == '':
            logger.tx.method = Method.FALLBACK
        return logger


    def jump_state(self, state_id):
        self.lib.JumpState(state_id)


    def set_balance(self, address, amount):
        params = {
            'address': str(address),
            'amount': str(amount),
        }
        params = json.dumps(params).encode('ascii')
        self.lib.SetBalance(params)