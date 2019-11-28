from ..obs_base import ObsBase
from ..symbolic import ObsSymbolic


class ObsMix(ObsBase):

    def __init__(self, contract_manager, account_manager, dataset_dump_path, backend_loggers):
        super().__init__(contract_manager, account_manager, dataset_dump_path)

        self.backend_loggers = backend_loggers
        self.obs_fuzz = ObsBase(contract_manager, account_manager, None)
        self.obs_symbolic = ObsSymbolic(contract_manager, account_manager, None, backend_loggers)
        self.sym_stat = self.obs_symbolic.sym_stat


    def init(self):
        super().init()
        self.obs_fuzz.init()
        self.obs_symbolic.init()


    def update(self, logger, is_init_txs):
        super().update(logger, is_init_txs)
        self.obs_fuzz.update(logger, is_init_txs)
        self.obs_symbolic.update(logger, is_init_txs)


    def update_fuzz(self, logger, is_init_txs):
        super().update(logger, is_init_txs)
        self.sym_stat.update(logger)
        self.obs_fuzz.update(logger, is_init_txs)


    def reset(self):
        self.obs_symbolic = ObsSymbolic(self.contract_manager, self.account_manager, None, self.backend_loggers)