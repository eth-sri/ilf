import json

from collections import OrderedDict


class IDGenerator:
    def __init__(self):
        self.ids = dict()

    def gen(self, contract):
        if contract not in self.ids:
            self.ids[contract] = 0

        idd = self.ids[contract]
        idd += 1
        self.ids[contract] = idd
        return idd

    def clear(self):
        self.ids.clear()


class Tx:

    IDGEN = IDGenerator()

    def __init__(self, policy, contract, call_address, method, input_bytes, arguments, amount, sender, timestamp, snapshot, idd=None):
        self.idd = Tx.IDGEN.gen(contract) if idd is None else idd

        if policy.__class__ == str:
            self.policy = policy
        else:
            self.policy = policy.__class__.__name__

        self.contract = contract
        self.call_address = call_address
        self.method = method
        self.input_bytes = input_bytes
        self.arguments = arguments
        self.amount = amount
        self.sender = sender
        self.timestamp = timestamp

        self.snapshot = snapshot


    def to_execution_dict(self):
        def recursive_to_str(args):
            for i in range(len(args)):
                if args[i].__class__ == list:
                    recursive_to_str(args[i])
                else:
                    args[i] = str(args[i])
        recursive_to_str(self.arguments)

        return {
            'idd': self.idd,
            'contract': self.contract,
            'call_address': self.call_address,
            'method': self.method,
            'input_bytes': list(self.input_bytes),
            'arguments': self.arguments,
            'amount': self.amount,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'snapshot': self.snapshot,
            'policy': self.policy,
        }


    def to_execution_str(self):
        return json.dumps(self.to_execution_dict())


    def to_json(self):
        j = OrderedDict()
        j['idd'] = self.idd
        j['contract'] = self.contract
        j['call_address'] = self.call_address
        j['method'] = self.method
        j['arguments'] = self.arguments
        j['amount'] = self.amount
        j['sender'] = self.sender
        j['timestamp'] = self.timestamp
        j['snapshot'] = self.snapshot
        j['policy'] = self.policy,
        return j