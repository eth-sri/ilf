import z3
from enum import Enum
import logging
import sys
import ethereum.abi
from ilf.symbolic.symbolic import svm_utils
from ilf.symbolic import utils
from ilf.symbolic.symbolic.storage import EmptyStorage, AbstractStorage
from collections import defaultdict
from copy import copy


class AccountType(Enum):
    DEFAULT = 1
    LIBRARY = 2

account_counter = 0

class Account:

    CONTRACT_TO_ACCOUNT_COUNT = defaultdict(int)

    def __init__(self,
                 address,
                 contract,
                 typ=AccountType.DEFAULT,
                 balance=None,
                 storage=None,
                 account_id=None,
                 mapping_id_to_sum=None):
        global account_counter
        if account_id is None:
            account_counter += 1
        self.id = account_id if account_id is not None else account_counter
        self.address = address
        self.contract = contract
        self.typ = typ
        self.balance = z3.BitVec(f'{address}_balance', 256) if balance is None else balance
        self.CONTRACT_TO_ACCOUNT_COUNT[contract] += 1
        self.contract_tag = contract.name + utils.ADDRESS_ARG_TAG + str(self.CONTRACT_TO_ACCOUNT_COUNT[contract])
        self.storage = storage if storage is not None else EmptyStorage()
        self.mapping_id_to_sum = mapping_id_to_sum if mapping_id_to_sum is not None else {}


    def abstract(self, label_suffix):
        old_storage = self.storage
        self.storage = AbstractStorage(f'abstract_storage{label_suffix}')
        for map_id in self.mapping_id_to_sum:
            if svm_utils.is_bv_concrete(map_id):
                map_id_string = svm_utils.get_concrete_int(map_id)
            else:
                map_id_string = str(z3.simplify(map_id))
                raise Exception("pdb")
            label = f'abstract_sum_{map_id_string}{label_suffix}'
            self.mapping_id_to_sum[map_id] = z3.BitVec(label, 256)
        self.balance = z3.BitVec(f'gstate_balance{label_suffix}', 256)

    def __deepcopy__(self, memo):
        return Account(self.address,
                       self.contract,
                       self.typ,
                       self.balance,
                       copy(self.storage),
                       self.id,
                       copy(self.mapping_id_to_sum))

    def __str__(self):
        return str({'Name': self.contract.name, 'id': self.id})

    def __repr__(self):
        return f'Account_{self.contract.name}_obj_id_{id(self)}_acc_id_{self.id}'
