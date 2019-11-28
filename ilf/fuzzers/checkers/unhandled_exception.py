from .checker import Checker
from ...ethereum import REVERT, INVALID


class UnhandledException(Checker):

    def __init__(self):
        super().__init__()


    def check(self, logger):
        has_exception = False

        for _, log in enumerate(logger.logs):
            if (log.op in (REVERT, INVALID) or log.error != '') and log.depth > 1:
                has_exception = True

        if has_exception and logger.logs[-1].op not in (REVERT, INVALID) and logger.logs[-1].error == '':
            return True
        else:
            return False