from .checker import Checker


class Leaking(Checker):

    def __init__(self):
        super().__init__()

    def check(self, logger):
        return 'leaking' in logger.bug_res