package main

import "C"
import (
	"encoding/json"
	"fmt"
	"ilf/execution"
)

var backend *execution.Backend

//export SetBackend
func SetBackend(path *C.char) *C.char {
	projPath := C.GoString(path)

	var fuzzLoggers []*execution.FuzzLogger
	backend, fuzzLoggers = execution.NewBackend(projPath)

	bs, err := json.Marshal(fuzzLoggers)
	if err != nil {
		panic(fmt.Errorf("error unmarshalling loggers"))
	}

	return C.CString(string(bs))
}

//export GetContracts
func GetContracts() *C.char {
	bs, err := json.Marshal(backend.Contracts)
	if err != nil {
		panic(fmt.Errorf("error marshalling contracts %v", err))
	}

	return C.CString(string(bs))
}

//export GetAccounts
func GetAccounts() *C.char {
	bs, err := json.Marshal(backend.Accounts)
	if err != nil {
		panic(fmt.Errorf("error marshalling accounts: %v", err))
	}

	return C.CString(string(bs))
}

//export JumpState
func JumpState(id int) {
	if snapshot, ok := backend.Snapshots[id]; ok {
		backend.StateDB = snapshot
		backend.Snapshots[id] = snapshot.Copy()
	}
}

//export CommitTx
func CommitTx(params *C.char) *C.char {
	paramsContent := []byte(C.GoString(params))

	tx := &execution.Tx{}
	err := json.Unmarshal(paramsContent, tx)
	if err != nil {
		panic(fmt.Errorf("error unmarshalling Tx"))
	}

	fuzzLogger := backend.CommitTx(tx)

	bs, err := json.Marshal(fuzzLogger)
	return C.CString(string(bs))
}

type SetBalanceParams struct {
	AddressStr string `json:"address"`
	AmountStr  string `json:"amount"`
}

//export SetBalance
func SetBalance(paramsStr *C.char) {
	paramsContent := []byte(C.GoString(paramsStr))

	params := &SetBalanceParams{}
	err := json.Unmarshal(paramsContent, params)
	if err != nil {
		panic(fmt.Errorf("error unmarshalling paramters for SetBalance"))
	}

	backend.SetBalance(params.AddressStr, params.AmountStr)
}

func main() {}
