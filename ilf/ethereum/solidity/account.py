class AccountManager:

    def __init__(self, *args, **kwargs):
        self.accounts = [Account(**account) for account in kwargs['accounts']]
        self.account_by_address = dict()
        self.attacker_indices = list()
        self.attacker_addresses = set()
        for i, account in enumerate(self.accounts):
            self.account_by_address[account.address] = account
            if account.is_attacker:
                self.attacker_addresses.add(account.address)
                self.attacker_indices.append(i)


    def __getitem__(self, index):
        return self.accounts[index]


    def __iter__(self):
        return iter(self.accounts)


class Account:

    def __init__(self, *args, **kwargs):
        self.address = kwargs['address']
        self.amount = kwargs['amount']
        self.is_attacker = kwargs['is_attacker']