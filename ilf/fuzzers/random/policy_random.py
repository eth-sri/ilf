import random

from ..policy_base import PolicyBase
from ...ethereum import SolType
from ...execution import Tx


class PolicyRandom(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager):
        super().__init__(execution, contract_manager, account_manager)


    def select_tx_for_method(self, contract, method, obs):
        self.slice_size = random.randint(1, 5)
        address = contract.addresses[0]
        sender = self._select_sender()
        arguments = self._select_arguments(contract, method, sender, obs)
        amount = self._select_amount(contract, method, sender, obs)
        timestamp = self._select_timestamp(obs)

        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        return tx


    def select_tx(self, obs):
        contract = self._select_contract()
        address = contract.addresses[0]
        method = self._select_method(contract)
        sender = self._select_sender()
        arguments = self._select_arguments(contract, method, sender, obs)
        amount = self._select_amount(contract, method, sender, obs)
        timestamp = self._select_timestamp(obs)

        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        return tx


    def _select_contract(self):
        contract_name = random.choice(self.contract_manager.fuzz_contract_names)
        return self.contract_manager[contract_name]


    def _select_method(self, contract):
        return random.choice(contract.abi.methods)


    def _select_sender(self):
        return random.choice(range(0, len(self.account_manager.accounts)))


    def _select_amount(self, contract, method, sender, obs):
        if sender in self.account_manager.attacker_indices:
            return 0

        if self.contract_manager.is_payable(contract.name, method.name):
            amount = random.randint(0, self.account_manager[sender].amount)
            return amount
        else:
            return 0


    def _select_arguments(self, contract, method, sender, obs):
        arguments = []
        for arg in method.inputs:
            t = arg.evm_type.t
            if t == SolType.IntTy:
                arguments.append(self._select_int(contract, method, arg.evm_type.size, obs))
            elif t == SolType.UintTy:
                arguments.append(self._select_uint(contract, method, arg.evm_type.size, obs))
            elif t == SolType.BoolTy:
                arguments.append(self._select_bool())
            elif t == SolType.StringTy:
                arguments.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arguments.append(self._select_slice(contract, method, sender, arg.evm_type.elem, obs))
            elif t == SolType.ArrayTy:
                arguments.append(self._select_array(contract, method, sender, arg.evm_type.size, arg.evm_type.elem, obs))
            elif t == SolType.AddressTy:
                arguments.append(self._select_address(sender))
            elif t == SolType.FixedBytesTy:
                arguments.append(self._select_fixed_bytes(arg.evm_type.size, obs))
            elif t == SolType.BytesTy:
                arguments.append(self._select_bytes(obs))
            else:
                assert False, 'type {} not supported'.format(t)
        return arguments


    def _select_int(self, contract, method, size, obs):
        p = 1 << (size - 1)
        return random.randint(-p, p-1)


    def _select_uint(self, contract, method, size, obs):
        p = 1 << size
        return random.randint(0, p-1)


    def _select_string(self, obs):
        bs = []
        size = random.randint(0, 40)
        for _ in range(size):
            bs.append(random.randint(1, 127))
        return bytearray(bs).decode('ascii')


    def _select_bool(self):
        return random.choice([True, False])


    def _select_slice(self, contract, method, sender, typ, obs):
        size = random.randint(1, 15)
        return self._select_array(contract, method, sender, size, typ, obs)


    def _select_array(self, contract, method, sender, size, typ, obs):
        t = typ.t
        arr = []
        for _ in range(size):
            if t == SolType.IntTy:
                arr.append(self._select_int(contract, method, typ.size, obs))
            elif t == SolType.UintTy:
                arr.append(self._select_uint(contract, method, typ.size, obs))
            elif t == SolType.BoolTy:
                arr.append(self._select_bool())
            elif t == SolType.StringTy:
                arr.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arr.append(self._select_slice(contract, method, sender, typ.elem, obs))
            elif t == SolType.ArrayTy:
                arr.append(self._select_array(contract, method, sender, typ.size, typ.elem, obs))
            elif t == SolType.AddressTy:
                arr.append(self._select_address(sender))
            elif t == SolType.FixedBytesTy:
                arr.append(self._select_fixed_bytes(typ.size, obs))
            elif t == SolType.BytesTy:
                arr.append(self._select_bytes(obs))
            else:
                assert False, 'type not supported'

        return arr


    def _select_address(self, sender):
        if sender in self.account_manager.attacker_indices:
            return random.choice(self.addresses)
        else:
            l = [addr for addr in self.addresses if addr not in self.account_manager.attacker_addresses]
            return random.choice(l)


    def _select_fixed_bytes(self, size, obs):
        bs = []
        for _ in range(size):
            bs.append(random.randint(0, 255))
        return bs


    def _select_bytes(self, obs):
        size = random.randint(1, 15)
        return self._select_fixed_bytes(size, obs)