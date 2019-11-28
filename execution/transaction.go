package execution

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"strconv"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
)

type MyTransaction struct {
	Hash             string `json:"hash"`
	AccountNonce     uint64 `json:"nonce"`
	BlockHash        string `json:"blockHash"`
	BlockNumber      int    `json:"blockNumber"`
	TransactionIndex int    `json:"transactionIndex"`
	From             string `json:"from"`
	Recipient        string `json:"to"`
	Amount           string `json:"value"`
	GasLimit         uint64 `json:"gas"`
	Price            string `json:"gasPrice"`
	Payload          string `json:"input"`
}

func ReadTransactions(path string, accounts *AccountManager) ([]*types.Transaction, []common.Address) {
	txJSONContent, err := os.Open(path)
	if err != nil {
		panic(fmt.Errorf("error opening transactions file %v: %v", path, err))
	}
	defer txJSONContent.Close()

	var txs []*types.Transaction
	var receipients []common.Address
	fileScanner := bufio.NewScanner(txJSONContent)
	for fileScanner.Scan() {
		t := MyTransaction{}
		err := json.Unmarshal([]byte(fileScanner.Text()), &t)
		if err != nil {
			panic(fmt.Errorf("error unmarshalling transaction: %v", err))
		}

		txs = append(txs, convertAndSign(&t, accounts))
		receipients = append(receipients, common.HexToAddress(t.Recipient))
	}

	// scanner can't read lines longer than 65536 characters
	if err := fileScanner.Err(); err != nil {
		panic(err)
	}

	return txs, receipients
}

func getPayload(data *string) []byte {
	dataStr := *data
	if dataStr[1] == 'x' {
		dataStr = dataStr[2:len(dataStr)]
	}

	payload, err := hex.DecodeString(dataStr)
	if err != nil {
		panic(fmt.Errorf("error decoding payload: %v", err))
	}

	return payload
}

func isContractCreation(t *MyTransaction) bool {
	if (t.Recipient == "") || (NullAddress == fmt.Sprintf("%x", common.HexToAddress(t.Recipient))) {
		return true
	}
	return false
}

func convertAndSign(t *MyTransaction, accounts *AccountManager) *types.Transaction {
	// convert amount and gasprice from hex to decimal
	amount, err := strconv.ParseInt(t.Amount, 10, 64)
	if err != nil {
		amount = 0
	}

	price, err := strconv.ParseInt(t.Price, 10, 64)
	if err != nil {
		price = 0
	}

	payload := getPayload(&t.Payload)
	var tx *types.Transaction
	if t.GasLimit > uint64(*MaxGasPool) {
		t.GasLimit = uint64(*MaxGasPool)
	}
	if isContractCreation(t) {
		tx = types.NewContractCreation(t.AccountNonce, big.NewInt(amount), t.GasLimit, big.NewInt(price), payload)
	} else {
		tx = types.NewTransaction(t.AccountNonce, common.HexToAddress(t.Recipient), big.NewInt(amount), t.GasLimit, big.NewInt(price), payload)
	}

	sender := common.HexToAddress(t.From)
	signed, err := types.SignTx(tx, types.HomesteadSigner{}, accounts.GetAccountFromAddress(sender).Key)
	if err != nil {
		panic(fmt.Errorf("error signing contract: %v", err))
	}

	return signed
}
