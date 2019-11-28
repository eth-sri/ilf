import abc


class Checker:

    def __init__(self):
        pass


    @abc.abstractmethod
    def check(self, logger):
        raise NotImplementedError