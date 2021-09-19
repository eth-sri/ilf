package execution

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/asm"
	"github.com/ethereum/go-ethereum/core/vm"
)

func getSolcVersion() int {
	out, _ := exec.Command("solc", "--version").Output()
	out_s := string(out)
	idx := strings.Index(out_s, "Version: 0.")
	if idx == -1 {
		return 4
	}
	out_s = out_s[idx+len("Version: 0."):]
	idx = strings.Index(out_s, ".")
	if idx == -1 {
		return 4
	}
	out_s = out_s[:idx]
	out_int, err := strconv.Atoi(out_s)
	if err != nil {
		return 4
	}
	return out_int
}

func getBzzr() string {
	bzzr := ""
	solcVersion := getSolcVersion()
	if solcVersion <= 4 {
		bzzr = fmt.Sprintf("%x", append(append([]byte{0xa1, 0x65}, []byte("bzzr0")...), []byte{0x58, 0x20}...))
	} else if solcVersion == 5 {
		bzzr = fmt.Sprintf("%x", append(append([]byte{0xa2, 0x65}, []byte("bzzr1")...), []byte{0x58, 0x20}...))
	} else {
		fmt.Fprintln(os.Stderr, "Warning: be carefule! ILF was not tested on solc version >= 0.6!")
		bzzr = fmt.Sprintf("%x", append(append([]byte{0xa2, 0x64}, []byte("ipfs")...), []byte{0x58, 0x22}...))
	}
	return bzzr
}

var bzzr0 = getBzzr()

type Contract struct {
	Name      string           `json:"name"`
	Addresses []common.Address `json:"addresses"`
	ABI       *abi.ABI         `json:"abi"`
	Payable   map[string]bool  `json:"payable"`
	Insns     []*Instruction   `json:"insns"`
}

func (contract *Contract) ContainAddress(address common.Address) bool {
	for _, contractAddress := range contract.Addresses {
		if contractAddress == address {
			return true
		}
	}
	return false
}

type ContractManager struct {
	DeployedContracts map[string]*Contract `json:"contracts"`
	ProjPath          string               `json:"proj_path"`
}

func (manager *ContractManager) GetContractByName(name string) *Contract {
	return manager.DeployedContracts[name]
}

type Instruction struct {
	PC  uint64    `json:"pc"`
	Arg *big.Int  `json:"arg"`
	Op  vm.OpCode `json:"op"`
}

func NewContract(
	name string,
	address common.Address,
	ABI *abi.ABI,
	payable map[string]bool,
	bytecode string,
) *Contract {
	idx := strings.Index(bytecode, bzzr0)
	bytecode = bytecode[:idx-1]
	script, _ := hex.DecodeString(replacePlaceHolders(bytecode))
	it := asm.NewInstructionIterator(script)

	var insns []*Instruction
	for it.Next() {
		arg := new(big.Int)
		insn := &Instruction{
			PC:  it.PC(),
			Arg: arg.SetBytes(it.Arg()),
			Op:  it.Op(),
		}
		insns = append(insns, insn)
	}

	contract := &Contract{
		Name:    name,
		ABI:     ABI,
		Payable: payable,
		Insns:   insns,
	}
	contract.Addresses = append(contract.Addresses, address)

	return contract
}

func GetSwarmHash(bytecode string) (string, bool) {
	idx := strings.Index(bytecode, bzzr0)
	if idx == -1 {
		return "", false
	}
	swarmHash := bytecode[idx+len(bzzr0) : idx+len(bzzr0)+64]
	return swarmHash, true
}

func replacePlaceHolders(bytecode string) string {
	var buffer bytes.Buffer
	for i := 0; i < len(bytecode); i++ {
		if bytecode[i] == '_' {
			i += 39
			buffer.WriteString("0000000000000000000000000000000000000000")
			continue
		}
		buffer.WriteString(string(bytecode[i]))
	}
	return buffer.String()
}
