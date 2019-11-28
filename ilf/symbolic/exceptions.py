class CompilerError(Exception):
    pass

class LTLParseError(Exception):
    pass

class DeploymentError(Exception):
    pass

class UnsatError(Exception):
    pass

class SVMRuntimeError(Exception):
    pass

class NoContractFoundError(Exception):
    pass
