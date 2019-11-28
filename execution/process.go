package execution

import (
	"fmt"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/crypto"
)

type ProcessOptions struct {
	DeployContracts   bool
	HashContracts     map[string]string
	DeployedBytecodes map[string]string
}

func (backend *Backend) ProcessLogger(logger *vm.StructLogger, tx *types.Transaction, options *ProcessOptions) map[string]bool {
	res := make(map[string]bool)

	structLogs := logger.StructLogs()
	for _, structLog := range structLogs {
		if options.DeployContracts {
			if options.HashContracts == nil || options.DeployedBytecodes == nil {
				panic(fmt.Errorf("wants to deploy contracts without ContractHashes or DeployedBytecodes provided"))
			}

			hash, found := GetSwarmHash(fmt.Sprintf("%x", structLog.Memory))
			if found {
				name := options.HashContracts[hash]
				address := getDeployedContractAddress(tx)
				backend.DeployContract(name, options.DeployedBytecodes[name], address)
			}
		}
	}

	return res
}

// TODO: handle case when contract call deployes multiple contracts
func getDeployedContractAddress(tx *types.Transaction) common.Address {
	from, err := types.Sender(types.HomesteadSigner{}, tx)
	if err != nil {
		panic(err)
	}
	return crypto.CreateAddress(from, tx.Nonce())
}
