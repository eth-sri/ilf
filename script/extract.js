module.exports = function(callback) {
    var fs = require('fs');
    var logFile = process.cwd() + `/transactions.json`;

    // remove log file
    fs.unlink(logFile, (err) => {
      if (!err) {
        console.log(logFile, ' was deleted');
      }
    });

    console.log(`creating file ${logFile}`);
    var block_num;
    web3.eth.getBlockNumber().then(function(result) {
      block_num = result;
      console.log("total number of blocks", block_num);

      // start from first block, block 0 is genesis, it doesn't contain any transaction
      for (var i = 1; i <= block_num; i++) {
        web3.eth.getBlock(i).then(function(block) {
          for (var idx = 0; idx < block.transactions.length; idx++) {
            web3.eth.getTransaction(block.transactions[idx]).then(function(transaction) {
              console.log("from: " + transaction.from + " to:" + transaction.to + " Nonce:" + transaction.nonce)
              transaction["gas"] = parseInt(transaction["gas"], 16)
              var tx_json = JSON.stringify(transaction) + "\n"
              fs.appendFileSync(logFile, tx_json);
        });
      }
        });
      }
    })
  }