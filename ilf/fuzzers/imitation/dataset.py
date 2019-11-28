import itertools
import json
import os
import random
from tqdm import tqdm


class GraphsCollection:

    def __init__(self):
        self.graph = {}

    def add_graph(self, contract, storage_args):
        if contract in self.graph:
            return self.graph[contract][1]

        edges = []
        num_fields = 0
        for store in storage_args:
            if 'SLOAD' in store and len(store['SLOAD']) > 0:
                num_fields = max(num_fields, max(store['SLOAD']) + 1)
            if 'SSTORE' in store and len(store['SSTORE']) > 0:
                num_fields = max(num_fields, max(store['SSTORE']) + 1)
        num_fields = min(num_fields, 100)

        for i, store in enumerate(storage_args):
            if 'SLOAD' in store:
                for j in store['SLOAD']:
                    if j < 100:
                        edges.append([j, i + num_fields])
            if 'SSTORE' in store:
                for j in store['SSTORE']:
                    if j < 100:
                        edges.append([i + num_fields, j])

        assert num_fields > 0 or len(storage_args) > 0
                        
        for i in range(num_fields + len(storage_args)):
            edges.append([i, i])
        self.graph[contract] = (num_fields, edges)
        return edges

    def get(self, contract):
        return self.graph[contract]


class Input:

    def __init__(self, contract, method_features, trace_op_bow, num_addresses):
        self.contract = contract
        self.method_features = method_features
        self.trace_op_bow = trace_op_bow
        self.num_addresses = num_addresses


class Output:

    def __init__(self, method_name, sender, addr_args, int_args, amount, use_train):
        self.method_name = method_name
        self.sender = sender
        self.addr_args = addr_args
        self.int_args = int_args
        self.amount = amount
        self.use_train = use_train


class Sample:

    def __init__(self, input, output):
        self.input = input
        self.output = output

    @staticmethod
    def get_inputs(samples):
        return [sample.input for sample in samples]


class Dataset:

    def __init__(self):
        self.data = []

    def add_sample(self, sample):
        self.data.append(sample)

    def size(self):
        return len(self.data)

    def inputs(self):
        return [sample.input for sample in self.data]

    def shuffle(self):
        random.shuffle(self.data)

    @staticmethod
    def split_train_valid(dataset, valid_percent):
        dataset.shuffle()
        n_valid = int(dataset.size() * valid_percent)

        train = Dataset()
        valid = Dataset()
        valid.data = dataset.data[:n_valid]
        train.data = dataset.data[n_valid:]
        assert train.size() + valid.size() == dataset.size()
        return train, valid

    def make_batches(self, batch_size):
        ret = []
        for i in range(0, len(self.data), batch_size):
            ret.append(self.data[i:min(i + batch_size, len(self.data))])
        return ret

    @staticmethod
    def load(dump_dir, addr_map, int_values, amounts):
        gc = GraphsCollection()
        dataset = Dataset()
        methods = {}

        best_cov, best_samples = {}, {}
        for filename in tqdm(os.listdir(dump_dir)):
            if not filename.endswith('.data'):
                continue

            tot_cov = 0
            tmp_samples = []
            with open(os.path.join(dump_dir, filename), 'r') as fin:
                method_op_bow = {}
                addresses = {}
                for line in fin:
                    d = json.loads(line)
                    if d['type'] == 'init':
                        assert len(d['contracts']) == 1
                        for contract, fmap in d['contracts'].items():
                            methods[contract] = sorted(fmap['methods'].keys())
                            addresses[contract] = [addr for addr in addr_map.keys()]
                            if 'addresses' in fmap:
                                addresses[contract] += fmap['addresses']
                            func_args = [fmap['methods'][method] for method in methods[contract]]
                            for method in methods[contract]:
                                method_op_bow[method] = fmap['methods'][method]['op_bow']
                            gc.add_graph(contract, func_args)
                    elif d['type'] == 'tx':
                        tx = d['tx']
                        target_method = tx['method']
                        sender = tx['sender']
                        contract = tx['contract']
                        amount = tx['amount']
                        policy = tx['policy'] if 'policy' in tx else None
                        tot_cov += d['insn_coverage_change']

                        try:
                            addr_args, int_args = [], []
                            for arg in tx['arguments']:
                                if arg in addresses[contract]:
                                    addr_args.append(addresses[contract].index(arg))
                                if isinstance(arg, int):
                                    if arg in int_values:
                                        int_args.append(int_values.index(arg))
                                    else:
                                        int_args.append(len(int_values))

                            all_features = {}
                            for method in method_op_bow:
                                all_features[method] = d['features']['methods'][method] + method_op_bow[method]
                        except KeyError: # remove this when bug is fixed
                            continue

                        use_train = True
                        # if d['insn_coverage_change'] < 0.0000001:
                        #     use_train = False
                        # if policy is not None and policy[0] == 'PolicyImitation':
                        #     use_train = False

                        if all_features[target_method][4] < 1.0 or (amount not in amounts):
                            amount = None
                        else:
                            amount = amounts.index(amount)
                        inp = Input(contract, all_features, d['trace_bow'], len(addresses[contract]))
                        out = Output(target_method, sender, addr_args, int_args, amount, use_train)
                        tmp_samples.append((d['insn_coverage_change'], Sample(inp, out)))
                    else:
                        raise ValueError('Wrong type')
            while len(tmp_samples) > 0 and tmp_samples[-1][0] < 0.0000001:
                tmp_samples = tmp_samples[:-1]

            # for inc_cov, sample in tmp_samples:
            #     print(inc_cov, sample.output.method_name, sample.output.sender, sample.output.addr_args, sample.output.int_args, sample.output.amount)
            # print('========')

            tmp_samples = [x[1] for x in tmp_samples]

            if filename not in best_samples:
                best_samples[filename] = []
            best_samples[filename].append((tot_cov, tmp_samples))

        for filename, samples in best_samples.items():
            best_samples[filename].sort(key=lambda x: -x[0])
            for cov, samples in best_samples[filename]:
                if len(samples) > 0:
                    dataset.add_sample(samples)

        print('Loaded dataset with %d samples' % dataset.size())
        return methods, gc, dataset
