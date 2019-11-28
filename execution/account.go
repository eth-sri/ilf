package execution

import (
	"crypto/ecdsa"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

var accountHexes = []string{
	"1c6dbb1fe61bbb7c256f0ffcbd34087e211173dbc8454220b8b166ed6ada5c00",
	"b1cff43bf95333788b080b6cd5c5e2fcbe321ccd4132ed80cb3e72478c69e9a7",
	"aa3eeb453426d9c9292f89be5fa7e6caa0330d312255f84c0caa6764ae1adf00",
	"34a5a824b045c9ce797589d334394c11ee28d9cd8757f1a9b0ccf0fd0008c641",
	"a7a163dcb33958498cf5736282f53e39bd6cb7a58f5d4a948445dc86faa34f90",
}

var accountAmount = "100000000000000000000000000000"

type Account struct {
	Key        *ecdsa.PrivateKey `json:"-"`
	Address    common.Address    `json:"address"`
	Amount     *big.Int          `json:"amount"`
	IsAttacker bool              `json:"is_attacker"`
}

type AccountManager struct {
	Accounts         []*Account                    `json:"accounts"`
	KeyToAccount     map[ecdsa.PrivateKey]*Account `json:"-"`
	AddressToAccount map[common.Address]*Account   `json:"-"`
}

func (manager *AccountManager) GetAccountFromAddress(address common.Address) *Account {
	return manager.AddressToAccount[address]
}

func (backend *Backend) InitAccountManager() {
	manager := &AccountManager{
		KeyToAccount:     make(map[ecdsa.PrivateKey]*Account),
		AddressToAccount: make(map[common.Address]*Account),
	}

	for i, hex := range accountHexes {
		key, _ := crypto.HexToECDSA(hex)
		amount := big.NewInt(0)
		amount, _ = amount.SetString(accountAmount, 10)
		account := &Account{
			Key:     key,
			Address: crypto.PubkeyToAddress(key.PublicKey),
			Amount:  amount,
		}
		backend.StateDB.SetBalance(account.Address, account.Amount)
		if i == 3 || i == 4 {
			account.IsAttacker = true
		}

		manager.Accounts = append(manager.Accounts, account)
		manager.KeyToAccount[*account.Key] = account
		manager.AddressToAccount[account.Address] = account
	}

	backend.Accounts = manager
}
