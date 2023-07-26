ILF: AI-based Fuzzer for Ethereum Smart Contracts <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================
<p align="center">
    <img width="500" alt="portfolio_view" src="https://www.sri.inf.ethz.ch/assets/images/ilf-logo-1.png">
</p>

ILF is an <ins>**I**</ins>mitation <ins>**L**</ins>earning based <ins>**F**</ins>uzzer for smart contracts. The fuzzing policy, which is used to generate transactions, is represented by an ensemble of neural networks and is learned from thousands of high-quality sequences of transactions generated using symbolic execution. ILF can be used to fuzz any Ethereum smart contract and outputs the coverage and a vulnerability report.

ILF is developed at [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/) as part of the [Machine Learning for Programming](https://www.sri.inf.ethz.ch/research/plml) and [Blockchain Security](https://www.sri.inf.ethz.ch/research/blockchain-security) projects. For mode details, please refer to [ILF CCS'19 paper](https://files.sri.inf.ethz.ch/website/papers/ccs19-ilf.pdf) and [slides](https://files.sri.inf.ethz.ch/website/slides/ccs19-ilf-slides.pdf).

## Setup

We provide a docker file, which we recommend to start with. To build and run:
```
$ docker build -t ilf .
$ docker run -it ilf
```

You can also follow the instructions in the Dockerfile to install ILF locally. If you experience build errors on Apple M chips, please refer to [#21](https://github.com/eth-sri/ilf/issues/21).

## Usage

### Fuzzing

To fuzz the example provided in the repo with ILF (the `imitation` fuzzing policy) using our pre-trained model in the `model` directory:
```
$ python3 -m ilf --proj ./example/crowdsale/ --contract Crowdsale --fuzzer imitation --model ./model/ --limit 2000
```
The `--fuzzer` argument can be replaced by:
* `random`: a uniformly random fuzzing policy.
* `symbolic`: a symbolic execution fuzzing policy based on depth first search of block states. This is used for generating training sequences.
* `sym_plus`: an augmentation of `symbolic` which can revisit encountered block states.
* `mix`: a fuzzing policy that randomly chooses `imitation` or `symbolic` for generating each transaction.

For fuzzing new contracts, one needs to provide a Truffle project (formatted as the example in `example/crowdsale`). Then the script `script/extract.py` should be called to extract deployment transactions of the contracts. For the example contract, the script runs as follows:
```
$ rm example/crowdsale/transactions.json
$ python3 script/extract.py --proj example/crowdsale/ --port 8545
```
Note that you need to kill existing `ganache-cli` processes listening the same port before calling this script.

### Training

For training, one needs to run `symbolic` on a set of training contracts to produce a dataset in a training directory. Usually tens of thousands of contracts are used for training. For demonstration purposes, we show how to produce a small training dataset from our example contract to the `train_data` directory:
```
$ mkdir train_data
$ python3 -m ilf --proj ./example/crowdsale/ --contract Crowdsale --limit 2000 --fuzzer symbolic --dataset_dump_path ./train_data/crowdsale.data
```

Run the scripts to select seed integer values and amount values from the training dataset, and put them into `ilf/fuzzers/imitation/int_values.py` and `ilf/fuzzers/imitation/amounts.py`, respectively:
```
$ python3 script/get_int_values.py --train_dir ./train_data
$ python3 script/get_amounts.py --train_dir ./train_data
```

Then the following command performs neural network training and outputs the trained networks in the `new_model` directory:
```
$ mkdir new_model
$ python3 -m ilf --fuzzer imitation --train_dir ./train_data --model ./new_model
```

### Automatically Constructing Truffle Projects

For evaluation and training purposes, one might want to automatically construct Truffle projects from a large set of contracts. To achieve this, one can write a script to automatically produce files required by Truffle projects, following the format in `example/crowdsale`. The compressed file `truffle_scripts.tar.gz` contains the scripts we used. Those scripts might not run directly but can give you a high level idea how things work.

## Citing ILF
```
@inproceedings{He:2019:LFS:3319535.3363230,
 author = {He, Jingxuan and Balunovi\'{c}, Mislav and Ambroladze, Nodar and Tsankov, Petar and Vechev, Martin},
 title = {Learning to Fuzz from Symbolic Execution with Application to Smart Contracts},
 booktitle = {Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security},
 series = {CCS '19},
 year = {2019},
 isbn = {978-1-4503-6747-9},
 location = {London, United Kingdom},
 pages = {531--548},
 numpages = {18},
 url = {http://doi.acm.org/10.1145/3319535.3363230},
 doi = {10.1145/3319535.3363230},
 acmid = {3363230},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {fuzzing, imitation learning, smart contracts, symbolic execution},
} 
```

## Contributors
* [Jingxuan He](https://www.sri.inf.ethz.ch/people/jingxuan)
* [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav)
* Nodar Ambroladze
* [Petar Tsankov](https://www.sri.inf.ethz.ch/people/petar)
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)
* Anton Permenev

## License and Copyright
* Copyright (c) 2019 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
