package execution

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/big"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus/ethash"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
)

const (
	NullAddress = "0000000000000000000000000000000000000000"
)

var (
	MaxGasPool = (new(core.GasPool).AddGas(8000000))
	CoinBase   = common.HexToAddress(NullAddress)
)

type Backend struct {
	BlockChain  *core.BlockChain
	StateDB     *state.StateDB
	ChainConfig *params.ChainConfig

	ProjPath  string
	Accounts  *AccountManager
	Contracts *ContractManager

	Snapshots map[int]*state.StateDB
}

func NewBackend(projPath string) (*Backend, []*FuzzLogger) {
	db, blockchain, err := core.ExpNewCanonical(ethash.NewFullFaker(), 1, true)
	if err != nil {
		panic(fmt.Errorf("error creating blockchain: %v", err))
	}
	statedb, _ := state.New(common.Hash{}, state.NewDatabase(db))
	chainConfig := &params.ChainConfig{
		ChainId:             big.NewInt(1),
		HomesteadBlock:      big.NewInt(0),
		DAOForkBlock:        nil,
		DAOForkSupport:      false,
		EIP150Block:         big.NewInt(0),
		EIP150Hash:          common.Hash{},
		EIP155Block:         big.NewInt(0),
		EIP158Block:         nil, //big.NewInt(0),
		ByzantiumBlock:      big.NewInt(0),
		ConstantinopleBlock: big.NewInt(0),
		Ethash:              new(params.EthashConfig),
		Clique:              nil,
	}

	backend := &Backend{
		BlockChain:  blockchain,
		StateDB:     statedb,
		ChainConfig: chainConfig,
		ProjPath:    projPath,
		Snapshots:   make(map[int]*state.StateDB),
	}

	backend.InitAccountManager()
	loggers := backend.DeployContracts()
	backend.Snapshots[0] = backend.StateDB.Copy()

	return backend, loggers
}

func (backend *Backend) DeployContracts() []*FuzzLogger {
	manager := &ContractManager{
		ProjPath: backend.ProjPath,
	}
	backend.Contracts = manager

	deployedBytecodes := backend.readDeployedBytecodes()
	hashContracts := backend.readHashContracts(deployedBytecodes)

	var fuzzLoggers []*FuzzLogger
	transactions, receipients := ReadTransactions(fmt.Sprintf("%v/transactions.json", backend.ProjPath), backend.Accounts)
	for i, transaction := range transactions {
		structLogger := backend.CommitTransaction(transaction, backend.getDefaultHeader(), &ProcessOptions{
			DeployContracts:   true,
			HashContracts:     hashContracts,
			DeployedBytecodes: deployedBytecodes,
		})

		tx := &Tx{
			CallAddress: receipients[i],
		}
		logger := &FuzzLogger{
			Tx:                   tx,
			Logs:                 structLogger.StructLogs(),
			ContractReceiveEther: false,
		}
		fuzzLoggers = append(fuzzLoggers, logger)
	}

	for _, account := range backend.Accounts.Accounts {
		backend.StateDB.SetBalance(account.Address, account.Amount)
	}

	return fuzzLoggers
}

func (backend *Backend) DeployContract(name string, bytecode string, address common.Address) {
	if name == "Migrations" {
		return
	}

	manager := backend.Contracts

	if manager.DeployedContracts == nil {
		manager.DeployedContracts = make(map[string]*Contract)
	}

	if contract, ok := manager.DeployedContracts[name]; ok {
		if !contract.ContainAddress(address) {
			contract.Addresses = append(contract.Addresses, address)
		}
	} else {
		ABI, payable, isLibrary := backend.getContractInfo(name)

		if isLibrary {
			return
		}

		manager.DeployedContracts[name] = NewContract(name, address, ABI, payable, bytecode)
	}
}

func (backend *Backend) getContractInfo(contractName string) (*abi.ABI, map[string]bool, bool) {
	type NodeJSON struct {
		ContractKind string `json:"contractKind,omitempty"`
		Name         string `json:"name,omitempty"`
	}
	type NodesJSON struct {
		Nodes []NodeJSON `json:"nodes"`
	}
	type ASTJSON struct {
		ContractName string    `json:"contractName"`
		AST          NodesJSON `json:"ast"`
	}
	type ABIJSON struct {
		ABI abi.ABI `json:"abi"`
	}

	abiPath := fmt.Sprintf("%v/build/contracts/%v.json", backend.ProjPath, contractName)
	abiBytes, err := ioutil.ReadFile(abiPath)
	if err != nil {
		panic(fmt.Errorf("error reading abi file %v: %v", abiPath, err))
	}

	dec := json.NewDecoder(strings.NewReader(string(abiBytes)))
	var ast ASTJSON
	if err := dec.Decode(&ast); err != nil {
		panic(fmt.Errorf("error processing contract %v ast: %v", contractName, err))
	}
	Libraries := make(map[string]bool)
	for _, node := range ast.AST.Nodes {
		if node.ContractKind == "library" {
			Libraries[node.Name] = true
		}
	}

	if _, ok := Libraries[contractName]; ok {
		return nil, nil, true
	}

	dec = json.NewDecoder(strings.NewReader(string(abiBytes)))
	var abiJSON ABIJSON
	if err := dec.Decode(&abiJSON); err != nil {
		panic(fmt.Errorf("error processing contract %v ABI: %v", contractName, err))
	}
	evmABI := abiJSON.ABI

	type MethodJSON struct {
		Name    string `json:"name"`
		Payable bool   `json:"payable"`
	}
	type MethodsJSON struct {
		Methods []MethodJSON `json:"abi"`
	}
	var methods MethodsJSON
	if err := json.Unmarshal(abiBytes, &methods); err != nil {
		panic(fmt.Errorf("error processing contract %v payble map: %v", contractName, err))
	}

	payable := make(map[string]bool)
	for _, method := range methods.Methods {
		payable[method.Name] = method.Payable
	}

	return &evmABI, payable, false
}

func (backend *Backend) readDeployedBytecodes() map[string]string {
	DeployedBytecodes := make(map[string]string)

	buildDir := fmt.Sprintf("%v/build/contracts/", backend.ProjPath)
	files, err := ioutil.ReadDir(buildDir)
	if err != nil {
		panic(fmt.Errorf("error reading truffle build directory %v: %v", buildDir, err))
	}

	for _, f := range files {
		contractFile := fmt.Sprintf("%v/build/contracts/%v", backend.ProjPath, f.Name())
		contractBytes, _ := ioutil.ReadFile(contractFile)
		var contractJSON map[string]string
		json.Unmarshal(contractBytes, &contractJSON)
		DeployedBytecodes[getFilename(f.Name())] = contractJSON["deployedBytecode"][2:]
	}

	return DeployedBytecodes
}

func (backend *Backend) readHashContracts(deployedBytecodes map[string]string) map[string]string {
	hashContracts := make(map[string]string)

	for contractName, byteCode := range deployedBytecodes {
		hash, found := GetSwarmHash(byteCode)
		if !found {
			log.Trace(fmt.Sprintf("swarm hash was not found in contract ABI: %v", contractName))
			continue
		}
		hashContracts[hash] = contractName
	}

	return hashContracts
}

func getFilename(filename string) string {
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

func (backend *Backend) getDefaultHeader() *types.Header {
	header := &types.Header{
		Coinbase:   CoinBase,
		ParentHash: backend.BlockChain.CurrentBlock().Hash(),
		Number:     big.NewInt(1),
		GasLimit:   math.MaxUint64,
		Difficulty: big.NewInt(int64(1)),
		Extra:      nil,
		Time:       big.NewInt(time.Now().Unix()),
	}
	return header
}

// TODO: geth does not support fucntion overloading
type Tx struct {
	ID          int            `json:"idd"`
	Contract    string         `json:"contract"`
	CallAddress common.Address `json:"call_address"`
	Method      string         `json:"method"`
	InputBytes  []int          `json:"input_bytes"`
	Arguments   []interface{}  `json:"arguments"`
	Amount      *big.Int       `json:"amount"`
	Sender      int            `json:"sender"`
	Timestamp   *big.Int       `json:"timestamp"`
	Snapshot    bool           `json:"snapshot"`
	Policy      string         `json:"policy"`
}

func (backend *Backend) CommitTransaction(
	transaction *types.Transaction,
	header *types.Header,
	options *ProcessOptions,
) *vm.StructLogger {
	snap := backend.StateDB.Snapshot()
	logconfig := &vm.LogConfig{
		DisableMemory: false,
		DisableStack:  false,
		Debug:         false,
	}
	logger := vm.NewStructLogger(logconfig)
	vmConfig := vm.Config{
		Debug:  true,
		Tracer: logger,
	}
	_, _, err := core.ApplyTransaction(
		backend.ChainConfig,
		backend.BlockChain,
		&CoinBase,
		(new(core.GasPool).AddGas(MaxGasPool.Gas())),
		backend.StateDB,
		header,
		transaction,
		&header.GasUsed,
		vmConfig,
	)

	if err != nil {
		backend.StateDB.RevertToSnapshot(snap)
		panic(fmt.Errorf("error when committing tx: %v", err))
	}

	backend.ProcessLogger(logger, transaction, options)

	return logger
}

type FuzzLogger struct {
	Tx                   *Tx             `json:"tx"`
	Logs                 []vm.StructLog  `json:"logs"`
	BugRes               map[string]bool `json:"bug_res"`
	ContractReceiveEther bool            `json:"contract_receive_ether"`
}

func (backend *Backend) SetBalance(addressStr string, amountStr string) {
	address := common.HexToAddress(addressStr)
	amount := big.NewInt(0)
	amount, err := amount.SetString(amountStr, 10)
	if !err {
		panic(fmt.Errorf("err unmarshalling %v to big int", amountStr))
	}

	backend.StateDB.SetBalance(address, amount)
}

func (backend *Backend) CommitTx(tx *Tx) *FuzzLogger {
	contract := backend.Contracts.GetContractByName(tx.Contract)

	method := tx.Method
	ABI := backend.Contracts.GetContractByName(tx.Contract).ABI
	methodABI := ABI.Methods[method]
	if len(methodABI.Inputs) != len(tx.Arguments) {
		panic(fmt.Errorf("length of arugments does not match ABI for contract %v and method %v", contract, method))
	}

	for i := 0; i < len(tx.Arguments); i++ {
		arg := formatArgsForTx(tx.Arguments[i], methodABI.Inputs[i].Type)
		tx.Arguments[i] = arg.Interface()
	}

	var input []byte
	if tx.Method != "" {
		var err error
		input, err = contract.ABI.Pack(tx.Method, tx.Arguments...)
		if err != nil {
			panic(fmt.Errorf("error packing arguments for method %v of contract %v: %v", tx.Method, tx.Contract, err))
		}
	} else {
		if tx.InputBytes != nil {
			for _, val := range tx.InputBytes {
				input = append(input, byte(val))
			}
		}
	}

	sender := backend.Accounts.Accounts[tx.Sender]

	transaction := types.NewTransaction(
		backend.StateDB.GetNonce(sender.Address),
		tx.CallAddress,
		tx.Amount,
		uint64(*MaxGasPool),
		big.NewInt(0),
		input,
	)

	signedTransaction, err := types.SignTx(transaction, types.HomesteadSigner{}, sender.Key)
	if err != nil {
		panic(fmt.Errorf("error signing tx for method %v of contract %v: %v", tx.Method, tx.Contract, err))
	}

	header := backend.getDefaultHeader()
	header.Time = tx.Timestamp

	contractOldBalance := backend.StateDB.GetBalance(tx.CallAddress)
	accountsOldBalances := make([]*big.Int, len(backend.Accounts.Accounts))
	for i, account := range backend.Accounts.Accounts {
		accountsOldBalances[i] = backend.StateDB.GetBalance(account.Address)
	}

	structLogger := backend.CommitTransaction(signedTransaction, header, &ProcessOptions{})

	contractNewBalance := backend.StateDB.GetBalance(tx.CallAddress)
	contractReceiveEther := contractNewBalance.Cmp(contractOldBalance) == 1

	attackerReceiveEther := false
	for i, account := range backend.Accounts.Accounts {
		newBalance := backend.StateDB.GetBalance(account.Address)
		if newBalance.Cmp(accountsOldBalances[i]) == 1 && account.IsAttacker {
			attackerReceiveEther = true
		}
	}

	bugRes := make(map[string]bool)
	if attackerReceiveEther {
		bugRes["leaking"] = true
	}

	for _, account := range backend.Accounts.Accounts {
		backend.StateDB.SetBalance(account.Address, account.Amount)
	}

	if tx.Snapshot {
		backend.Snapshots[tx.ID] = backend.StateDB.Copy()
	}

	fuzzLogger := &FuzzLogger{
		Tx:                   tx,
		Logs:                 structLogger.StructLogs(),
		BugRes:               bugRes,
		ContractReceiveEther: contractReceiveEther,
	}

	return fuzzLogger
}

func formatArgsForTx(arg interface{}, typ abi.Type) reflect.Value {
	var ret reflect.Value

	switch typ.T {
	case abi.IntTy:
		argStr := arg.(string)
		switch typ.Size {
		case 8, 16, 32, 64:
			value, err := strconv.ParseInt(argStr, 10, typ.Size)
			if err != nil {
				panic(fmt.Errorf("error parsing string %v to int%v", argStr, typ.Size))
			}
			switch typ.Size {
			case 8:
				ret = reflect.ValueOf(int8(value))
			case 16:
				ret = reflect.ValueOf(int16(value))
			case 32:
				ret = reflect.ValueOf(int32(value))
			case 64:
				ret = reflect.ValueOf(value)
			}
		default:
			value := new(big.Int)
			value, ok := value.SetString(argStr, 10)
			if !ok {
				panic(fmt.Errorf("error parsing string %v to big int", argStr))
			}
			ret = reflect.ValueOf(value)
		}
	case abi.UintTy:
		argStr := arg.(string)
		switch typ.Size {
		case 8, 16, 32, 64:
			value, err := strconv.ParseUint(argStr, 10, typ.Size)
			if err != nil {
				panic(fmt.Errorf("error parsing string %v to uint%v", argStr, typ.Size))
			}
			switch typ.Size {
			case 8:
				ret = reflect.ValueOf(uint8(value))
			case 16:
				ret = reflect.ValueOf(uint16(value))
			case 32:
				ret = reflect.ValueOf(uint32(value))
			case 64:
				ret = reflect.ValueOf(value)
			}
		default:
			value := new(big.Int)
			value, ok := value.SetString(argStr, 10)
			if !ok {
				panic(fmt.Errorf("error parsing string %v to big int", argStr))
			}
			ret = reflect.ValueOf(value)
		}
	case abi.BoolTy:
		argStr := arg.(string)
		value, err := strconv.ParseBool(argStr)
		if err != nil {
			panic(fmt.Errorf("error parsing string %v to bool type", argStr))
		}
		ret = reflect.ValueOf(value)
	case abi.StringTy:
		argStr := arg.(string)
		value := argStr
		ret = reflect.ValueOf(value)
	case abi.AddressTy:
		argStr := arg.(string)
		value := common.HexToAddress(argStr)
		ret = reflect.ValueOf(value)
	case abi.SliceTy, abi.BytesTy:
		value := reflect.New(typ.Type).Elem()
		argSlice := reflect.ValueOf(arg)
		for i := 0; i < argSlice.Len(); i++ {
			var elem reflect.Value
			if typ.T == abi.BytesTy {
				elemStr := argSlice.Index(i).Interface().(string)
				elemValue, err := strconv.ParseUint(elemStr, 10, 8)
				if err != nil {
					panic(fmt.Errorf("error parsing string %v into uint8", elemStr))
				}
				elem = reflect.ValueOf(byte(elemValue))
			} else {
				elem = formatArgsForTx(argSlice.Index(i).Interface(), *typ.Elem)
			}
			value = reflect.Append(value, elem)
		}

		ret = value
	case abi.ArrayTy, abi.FixedBytesTy:
		value := reflect.New(typ.Type).Elem()
		argArray := reflect.ValueOf(arg)
		if typ.Size != argArray.Len() {
			panic(fmt.Errorf("array sizes do not match between concrete arguments and abi"))
		}

		for i := 0; i < argArray.Len(); i++ {
			var elem reflect.Value
			if typ.T == abi.FixedBytesTy {
				elemStr := argArray.Index(i).Interface().(string)
				elemValue, err := strconv.ParseUint(elemStr, 10, 8)
				if err != nil {
					panic(fmt.Errorf("error parsing string %v into uint8", elemStr))
				}
				elem = reflect.ValueOf(byte(elemValue))
			} else {
				elem = formatArgsForTx(argArray.Index(i).Interface(), *typ.Elem)
			}
			value.Index(i).Set(elem)
		}

		ret = value
	default:
		panic(fmt.Errorf("unsupported type: %v", typ.T))
	}

	return ret
}
