import time
import random
import numpy as np
import itertools
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from enum import Enum
from tqdm import tqdm
from ..policy_base import PolicyBase
from ...ethereum import SolType
from ...execution import Tx
from .models import PolicyNet, EmbedGCN, ParamsNet, ArgsNet
from .dataset import Input, Sample, Dataset, GraphsCollection
from .constants import BOW_SIZE, GCN_HIDDEN, MINI_BATCH_SIZE, ADAM_LEARNING_RATE, NUM_EPOCHS
from .nlp import NLP
from .int_values import INT_VALUES
from .amounts import AMOUNTS
from .addr_map import ADDR_MAP


use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(use_cuda)

ADDR_FEAT = 10
RNN_HIDDEN_SIZE = 100
NUM_LAYERS = 1
RAW_FEATURE_SIZE = 65 + 300
INT_EXPLORE_RATE = -1


class PolicyImitation(PolicyBase):

    def __init__(self, execution, contract_manager, account_manager, args):
        if contract_manager is not None:
            super().__init__(execution, contract_manager, account_manager)

        self.addr_map = ADDR_MAP
        self.int_values = INT_VALUES
        self.amounts = AMOUNTS
        self.slice_size = 2

        self.raw_feature_size = RAW_FEATURE_SIZE
        self.feature_size = RNN_HIDDEN_SIZE
        self.state_size = RNN_HIDDEN_SIZE

        self.net = PolicyNet(self.raw_feature_size, self.feature_size, self.state_size).to(device)
        self.gcn = EmbedGCN(self.feature_size, GCN_HIDDEN, self.feature_size).to(device)
        self.params_net = ParamsNet(RNN_HIDDEN_SIZE).to(device)
        self.addr_args_net = ArgsNet(10, RNN_HIDDEN_SIZE).to(device)
        self.int_args_net = ArgsNet(len(self.int_values)+1, RNN_HIDDEN_SIZE).to(device)
        self.rnn = nn.GRU(self.feature_size, RNN_HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(device)
        self.scaler = None
        
        self.graphs_col = GraphsCollection()
        self.last_method = dict()
        self.hidden = dict()
        self.first_hidden = dict()
        self.graph_embeddings = dict()
        self.args = args
        self.method_names = {}
        self.method_bows = {}
        self.nlp = NLP()
        self.nlp.w2v = pickle.load(open('ilf_w2v.pkl', 'rb'))

        self.adam = Adam(list(self.net.parameters()) + list(self.params_net.parameters()) + list(self.gcn.parameters()) + \
                    list(self.int_args_net.parameters()) + list(self.addr_args_net.parameters()) + list(self.rnn.parameters()),
                    lr=ADAM_LEARNING_RATE,
                    weight_decay=1e-5)
        self.scheduler = StepLR(self.adam, step_size=1000, gamma=1.0)


    def start_train(self):
        print('starting training from {} of dataset'.format(self.args.train_dir))
        method_names, gc, dataset = Dataset.load(self.args.train_dir, self.addr_map, self.int_values, self.amounts)
        self.graphs_col = gc
        self.method_names = method_names
        self.train(dataset)


    def load_model(self):
        load_dir = self.args.model
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        if use_cuda == 'cuda':
            self.net.load_state_dict(torch.load(os.path.join(load_dir, 'net.pt')))
            self.gcn.load_state_dict(torch.load(os.path.join(load_dir, 'gcn.pt')))
            self.params_net.load_state_dict(torch.load(os.path.join(load_dir, 'params_net.pt')))
            self.addr_args_net.load_state_dict(torch.load(os.path.join(load_dir, 'addr_args_net.pt')))
            self.int_args_net.load_state_dict(torch.load(os.path.join(load_dir, 'int_args_net.pt')))
            self.rnn.load_state_dict(torch.load(os.path.join(load_dir, 'rnn.pt')))
        else:
            self.net.load_state_dict(torch.load(os.path.join(load_dir, 'net.pt'), map_location='cpu'))
            self.gcn.load_state_dict(torch.load(os.path.join(load_dir, 'gcn.pt'), map_location='cpu'))
            self.params_net.load_state_dict(torch.load(os.path.join(load_dir, 'params_net.pt'), map_location='cpu'))
            self.addr_args_net.load_state_dict(torch.load(os.path.join(load_dir, 'addr_args_net.pt'), map_location='cpu'))
            self.int_args_net.load_state_dict(torch.load(os.path.join(load_dir, 'int_args_net.pt'), map_location='cpu'))
            self.rnn.load_state_dict(torch.load(os.path.join(load_dir, 'rnn.pt'), map_location='cpu'))
        self.net.eval()
        self.gcn.eval()
        self.params_net.eval()
        self.addr_args_net.eval()
        self.int_args_net.eval()
        self.rnn.eval()

    def calc_method_features(self, contract_name, method_features, scale=False):
        num_methods = len(self.method_names[contract_name])
        features = np.zeros((num_methods, self.raw_feature_size))
        for i, method in enumerate(self.method_names[contract_name]):
            method_w2v = self.nlp.embed_method(method)
            method_feats = np.concatenate([np.array(method_features[method]), method_w2v], axis=0)
            features[i, :self.raw_feature_size] = method_feats[:self.raw_feature_size]
        if scale:
            features = self.scaler.transform(features)
        return features

    def compute_init_hidden(self, inputs):
        big_edges, big_features = [], []
        off = 0
        for input in inputs:
            num_methods = len(self.method_names[input.contract])

            raw_method_features = input.method_features
            features = self.calc_method_features(input.contract, input.method_features, True)
            num_fields, edges = self.graphs_col.get(input.contract)

            big_features.append(np.zeros((num_fields, self.raw_feature_size)))
            big_features.append(features)

            edges = torch.LongTensor(edges).to(device) + off
            big_edges.append(edges)
            off += num_methods + num_fields
        big_edges = torch.cat(big_edges, dim=0)
            
        big_features = np.concatenate(big_features, axis=0)
        big_features = torch.from_numpy(big_features).float().to(device)
        comp_features = self.net.compress_features(big_features)

        if use_cuda == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(
                big_edges.t(),
                torch.ones(big_edges.size()[0]).to(device),
                torch.Size([off, off]))
        else:
            adj = torch.sparse.FloatTensor(
                big_edges.t(),
                torch.ones(big_edges.size()[0]).to(device),
                torch.Size([off, off]))
        
        graph_embeddings = self.gcn(comp_features, adj)

        all_state_feat = []
        off = 0
        for input in inputs:
            num_fields = self.graphs_col.get(input.contract)[0]
            num_methods = len(self.method_names[input.contract])
            idx_b, idx_e = off, off + num_fields + num_methods
            all_state_feat.append(torch.mean(graph_embeddings[idx_b:idx_e], dim=0).unsqueeze(0))
            off += num_methods + num_fields
        assert off == graph_embeddings.size()[0]
        return torch.cat(all_state_feat, dim=0), graph_embeddings

    def compute_rnn_inputs(self, batch, graph_embeddings):
        seq_len = 0
        for samples in batch:
            seq_len = max(seq_len, len(samples))
        inp = np.zeros((seq_len, len(batch), self.feature_size))
        for i, samples in enumerate(batch):
            for j in range(len(samples) - 1):
                next_feats = self.calc_method_features(samples[j+1].input.contract, 
                                                       samples[j+1].input.method_features,
                                                       True)
                next_feats = torch.from_numpy(next_feats).float().to(device)
                next_feats = self.net.compress_features(next_feats)
                target_idx = self.method_names[samples[j].input.contract].index(samples[j].output.method_name)
                inp[j + 1, i, :] = next_feats[target_idx].detach().cpu().numpy()
        return torch.from_numpy(inp).float().to(device)
    
    def compute_f(self, batch, rnn_out, graph_embeddings):
        x_state, x_method_feat, x_method_graph = [], [], []
        off = 0
        for i, samples in enumerate(batch):
            for j, sample in enumerate(samples):
                num_fields = self.graphs_col.get(sample.input.contract)[0]
                num_methods = len(sample.input.method_features)
                feats = self.calc_method_features(sample.input.contract, 
                                                  sample.input.method_features,
                                                  True)
                x_method_feat.append(torch.from_numpy(feats).float().to(device))
                x_method_graph.append(graph_embeddings[off+num_fields:off+num_fields+num_methods])
                x_state.append(rnn_out[j, i].view(1, -1).repeat(num_methods, 1))
            off += num_fields + num_methods
        assert off == graph_embeddings.size()[0]
        
        x_method_feat = torch.cat(x_method_feat, dim=0).to(device)
        x_method_graph = torch.cat(x_method_graph, dim=0).to(device)
        x_method_feat = self.net.compress_features(x_method_feat)
        x_method = torch.cat([x_method_feat, x_method_graph], dim=1)
        x_state = torch.cat(x_state, dim=0)
        f_outs = self.net.predict_method(x_state, x_method)
        return f_outs

    def compute_sender_amount(self, batch, rnn_out):
        x_feat = []
        for i, samples in enumerate(batch):
            for j, sample in enumerate(samples):
                x_feat.append(rnn_out[j, i].view(1, -1))
        x_feat = torch.cat(x_feat, dim=0).to(device)
        sender_outs = self.params_net.predict_sender(x_feat)
        amount_outs = self.params_net.predict_amount(x_feat)
        return sender_outs, amount_outs

    def compute_addr_args(self, batch, rnn_out):
        hidden, max_args = [], 0
        for i, samples in enumerate(batch):
            for j, sample in enumerate(samples):
                hidden.append(rnn_out[j, i].view(1, -1))
                max_args = max(max_args, len(sample.output.addr_args))
        hidden = torch.cat(hidden, dim=0)
        input = torch.zeros((hidden.size()[0], 10)).to(device)

        addr_outs = []
        for idx in range(max_args):
            out, hidden = self.addr_args_net(input, hidden)
            addr_outs.append(out)
            
            input = torch.zeros((hidden.size()[0], 10)).to(device)
            curr_idx = 0
            for i, samples in enumerate(batch):
                for j, sample in enumerate(samples):
                    if idx < len(sample.output.addr_args) and sample.output.addr_args[idx] < 10:
                        input[curr_idx, sample.output.addr_args[idx]] = 1
                    curr_idx += 1
            assert curr_idx == hidden.size()[0]
        return addr_outs

    def compute_int_args(self, batch, rnn_out):
        hidden, max_args = [], 0
        for i, samples in enumerate(batch):
            for j, sample in enumerate(samples):
                hidden.append(rnn_out[j, i].view(1, -1))
                max_args = max(max_args, len(sample.output.int_args))
        hidden = torch.cat(hidden, dim=0)
        input = torch.zeros((hidden.size()[0], len(self.int_values)+1)).to(device)

        int_outs = []
        for idx in range(max_args):
            out, hidden = self.int_args_net(input, hidden)
            int_outs.append(out)
            
            input = torch.zeros((hidden.size()[0], len(self.int_values)+1)).to(device)
            curr_idx = 0
            for i, samples in enumerate(batch):
                for j, sample in enumerate(samples):
                    if idx < len(sample.output.int_args):
                        input[curr_idx, sample.output.int_args[idx]] = 1
                    curr_idx += 1
            assert curr_idx == hidden.size()[0]
        return int_outs
    

    def evaluate(self, dataset, epoch):
        batches = dataset.make_batches(MINI_BATCH_SIZE)
        tot_loss, tot_amount_loss, tot_facc, tot_sacc, tot_addr_acc, tot_int_acc, tot_amount_acc = 0, 0, 0, 0, 0, 0, 0
        tot_amount = 0
        
        assert len(batches) > 0

        for batch in tqdm(batches):
            init = []
            for samples in batch:
                init.append(samples[0].input)
            init_hidden = torch.zeros((NUM_LAYERS, len(batch), RNN_HIDDEN_SIZE)).to(device)
            first_hidden, graph_embeddings = self.compute_init_hidden(init)
            init_hidden[0] = first_hidden

            inp = self.compute_rnn_inputs(batch, graph_embeddings)
            rnn_out, _ = self.rnn(inp, init_hidden)
            f_outs = self.compute_f(batch, rnn_out, graph_embeddings)
            sender_outs, amount_outs = self.compute_sender_amount(batch, rnn_out)
            addr_outs = self.compute_addr_args(batch, rnn_out)
            int_outs = self.compute_int_args(batch, rnn_out)

            batch_loss, amount_loss, batch_facc, batch_sacc, batch_addr_acc, batch_int_acc = 0, 0, 0, 0, 0, 0
            off, num_samples, num_pred_addr, num_pred_int, num_amount = 0, 0, 0, 0, 0

            # method
            off, idx = 0, 0
            for i, samples in enumerate(batch):
                for j, sample in enumerate(samples):
                    num_methods = len(sample.input.method_features)
                    num_addresses = sample.input.num_addresses
                    if not sample.output.use_train:
                        off += num_methods
                        idx += 1
                        continue
                    
                    f_log_probs = F.log_softmax(f_outs[off:off+num_methods].view(-1), dim=0)
                    sender_log_probs = F.log_softmax(sender_outs[idx], dim=0)
                    amount_log_probs = F.log_softmax(amount_outs[idx], dim=0)
                    
                    addr_loss, addr_acc = 0, 0
                    for k in range(len(sample.output.addr_args)):
                        addr_log_probs = F.log_softmax(addr_outs[k][idx][:num_addresses], dim=0)
                        _, pred_addr = torch.max(addr_log_probs, dim=0)
                        addr_acc += 1 if pred_addr == sample.output.addr_args[k] else 0
                        addr_loss += -addr_log_probs[sample.output.addr_args[k]]
                        num_pred_addr += 1
                    if len(sample.output.addr_args) > 0:
                        addr_loss /= len(sample.output.addr_args)
                        batch_loss += addr_loss
                    
                    int_loss, int_acc = 0, 0
                    for k in range(len(sample.output.int_args)):
                        if sample.output.int_args[k] == len(self.int_values):
                            continue
                        int_log_probs = F.log_softmax(int_outs[k][idx], dim=0)
                        _, pred_int = torch.max(int_log_probs, dim=0)
                        int_acc += 1 if pred_int == sample.output.int_args[k] else 0
                        int_loss += -int_log_probs[sample.output.int_args[k]]
                        num_pred_int += 1
                    if len(sample.output.int_args) > 0:
                        int_loss /= len(sample.output.int_args)
                        batch_loss += int_loss

                    target_f = self.method_names[sample.input.contract].index(sample.output.method_name)
                    target_sender = sample.output.sender
                    target_amount = sample.output.amount
                    _, pred_f = torch.max(f_log_probs, dim=0)
                    _, pred_sender = torch.max(sender_log_probs, dim=0)
                    _, pred_amount = torch.max(amount_log_probs, dim=0)
                    if target_amount is not None:
                        batch_loss += -amount_log_probs[target_amount]
                        tot_amount += 1
                        tot_amount_acc += 1 if pred_amount == target_amount else 0

                    batch_loss += -f_log_probs[target_f]
                    batch_loss += -sender_log_probs[target_sender]
                    batch_facc += 1 if pred_f == target_f else 0
                    batch_sacc += 1 if pred_sender == target_sender else 0
                    batch_addr_acc += addr_acc
                    batch_int_acc += int_acc
                    
                    num_samples += 1
                    off += num_methods
                    idx += 1
            assert off == f_outs.size()[0]
            assert idx == sender_outs.size()[0]

            batch_loss /= num_samples
            batch_facc /= num_samples
            batch_sacc /= num_samples
            if num_pred_addr > 0:
                batch_addr_acc /= num_pred_addr
            if num_pred_int > 0:
                batch_int_acc /= num_pred_int

            self.adam.zero_grad()
            batch_loss.backward()
            self.adam.step()
            
            tot_loss += batch_loss.item()
            tot_amount_loss += amount_loss
            tot_facc += batch_facc
            tot_sacc += batch_sacc
            tot_addr_acc += batch_addr_acc
            tot_int_acc += batch_int_acc
        
        tot_loss /= len(batches)
        tot_amount_loss /= len(batches)
        tot_facc /= len(batches)
        tot_sacc /= len(batches)
        tot_addr_acc /= len(batches)
        tot_int_acc /= len(batches)
        tot_amount_acc /= tot_amount
        
        return tot_loss, tot_amount_loss, tot_facc, tot_sacc, tot_addr_acc, tot_int_acc, tot_amount_acc

    def train(self, train, valid=None):
        self.net.train()
        self.gcn.train()
        self.params_net.train()
        self.addr_args_net.train()
        self.rnn.train()

        savedir = self.args.model
        print('saving to ', savedir)
        try:
            os.makedirs(savedir)
        except FileExistsError:
            print('Warning: directory {} already exists, it will be overwritten!'.format(savedir))

        prev_best_facc = None

        all_feat = []
        for samples in train.data:
            for sample in samples:
                features = self.calc_method_features(sample.input.contract, sample.input.method_features)
                all_feat.append(features)
        all_feat = np.concatenate(all_feat, axis=0)
        if all_feat.shape[0] > 5000:
            all_feat = all_feat[:5000]
        self.scaler = StandardScaler()
        self.scaler.fit(all_feat)
        joblib.dump(self.scaler, os.path.join(savedir, 'scaler.pkl'))

        for epoch in range(NUM_EPOCHS):
            self.scheduler.step()
            train.shuffle()
            tot_loss, _, tot_facc, tot_sacc, tot_addr_acc, tot_int_acc, tot_amount_acc = self.evaluate(train, epoch)
            if epoch % 1 == 0:
                print('[TRAIN] Epoch = %d, Loss = %.4f, Acc@F = %.2lf, Acc@S = %.2lf, Acc@ADDR = %.2lf, Acc@INT = %.2lf, Acc@AMO = %.2lf' % (
                    epoch, tot_loss, tot_facc*100, tot_sacc*100, tot_addr_acc*100, tot_int_acc*100, tot_amount_acc*100))
            if epoch % 1 == 0:
                torch.save(self.net.state_dict(), os.path.join(savedir, 'net_{}.pt'.format(epoch)))
                torch.save(self.params_net.state_dict(), os.path.join(savedir, 'params_net_{}.pt'.format(epoch)))
                torch.save(self.gcn.state_dict(), os.path.join(savedir, 'gcn_{}.pt'.format(epoch)))
                torch.save(self.addr_args_net.state_dict(), os.path.join(savedir, 'addr_args_net_{}.pt'.format(epoch)))
                torch.save(self.int_args_net.state_dict(), os.path.join(savedir, 'int_args_net_{}.pt'.format(epoch)))
                torch.save(self.rnn.state_dict(), os.path.join(savedir, 'rnn_{}.pt'.format(epoch)))


    def clear_history(self):
        for contract in self.hidden:
            self.hidden[contract][0] = self.first_hidden[contract]


    def select_method(self, contract, obs):
        method_feats = {}
        for m in contract.abi.methods:
            self.method_bows[m.name] = m.bow
        for method, feats in obs.record_manager.get_method_features(contract.name).items():
            method_feats[method] = feats + self.method_bows[method]
        trace_op_bow = obs.trace_bow
        curr_input = Input(contract.name, method_feats, trace_op_bow, len(self.addresses))
        
        if contract.name not in self.last_method:
            self.graphs_col.add_graph(contract.name, [m.storage_args for m in contract.abi.methods])
            self.method_names[contract.name] = [m.name for m in contract.abi.methods]
            # self.first_tx = False

            self.hidden[contract.name] = torch.zeros((NUM_LAYERS, 1, RNN_HIDDEN_SIZE)).to(device)
            first_hidden, self.graph_embeddings[contract.name] = self.compute_init_hidden([curr_input])
            self.first_hidden[contract.name] = first_hidden
            self.hidden[contract.name][0] = first_hidden

        if contract.name not in self.last_method:
            rnn_input = torch.zeros((1, 1, self.feature_size)).to(device)
        else:
            rnn_input = self.calc_method_features(contract.name, method_feats, True)
            rnn_input = torch.from_numpy(rnn_input[self.last_method[contract.name]]).float().to(device)
            rnn_input = self.net.compress_features(rnn_input)
            rnn_input = rnn_input.view((1, 1, self.feature_size))

        rnn_out, self.hidden[contract.name] = self.rnn(rnn_input, self.hidden[contract.name])
        sample = Sample(curr_input, None)
        
        f_outs = self.compute_f([[sample]], rnn_out, self.graph_embeddings[contract.name]).view(-1)
        f_probs = F.softmax(f_outs, dim=0).detach().cpu().numpy()
        pred_f = np.random.choice(len(contract.abi.methods), p=f_probs)

        return pred_f, sample, rnn_out

    def update_tx(self, tx, obs):
        # contract = self._select_contract()
        contract = self.contract_manager[tx.contract]
        self.select_method(contract.name, obs)
        self.last_method[contract.name] = self.method_names[contract.name].index(tx.method)

    def select_tx(self, obs):
        r = random.random()
        if r >= 0.2:
            self.slice_size = random.randint(1, 5)
        else:
            self.slice_size = None
        contract = self._select_contract()
        address = contract.addresses[0]

        pred_f, sample, rnn_out = self.select_method(contract, obs)

        sender_outs, amount_outs = self.compute_sender_amount([[sample]], rnn_out)
        sender_probs = F.softmax(sender_outs.view(-1), dim=0).detach().cpu().numpy()
        amount_probs = F.softmax(amount_outs.view(-1), dim=0).detach().cpu().numpy()
        pred_sender = np.random.choice(len(self.addr_map), p=sender_probs)
        pred_amount = np.random.choice(len(self.amounts), p=amount_probs)

        method = contract.abi.methods[pred_f]

        attacker_indices = self.account_manager.attacker_indices
        if np.random.random() < len(attacker_indices) / len(self.account_manager.accounts):
            sender = int(np.random.choice(attacker_indices))
        else:
            sender = pred_sender
        arguments, addr_args, int_args = self._select_arguments(contract, method, sender, obs, rnn_out)
        amount = self._select_amount(contract, method, sender, obs, pred_amount)
        timestamp = self._select_timestamp(obs)

        self.last_method[contract.name] = pred_f

        tx = Tx(self, contract.name, address, method.name, bytes(), arguments, amount, sender, timestamp, True)
        return tx

    def _select_contract(self):
        contract_name = random.choice(self.contract_manager.fuzz_contract_names)
        return self.contract_manager[contract_name]

    def _select_amount(self, contract, method, sender, obs, pred_amount=None):
        if sender in self.account_manager.attacker_indices:
            return 0

        if self.contract_manager.is_payable(contract.name, method.name):
            if pred_amount is None:
                amount = random.randint(0, self.account_manager[sender].amount)
            else:
                amount = self.amounts[pred_amount]
            return amount
        else:
            return 0

    def _select_sender(self):
        return random.choice(range(0, len(self.account_manager.accounts)))

    def _select_arguments(self, contract, method, sender, obs, rnn_out):
        hidden_addr = rnn_out[0, 0].view(1, -1)
        input_addr = torch.zeros((1, 10)).to(device)
        hidden_int = rnn_out[0, 0].view(1, -1)
        input_int = torch.zeros((1, len(self.int_values)+1)).to(device)

        arguments, addr_args, int_args = [], [], []
        for arg in method.inputs:
            t = arg.evm_type.t
            if t == SolType.IntTy or t == SolType.UintTy:
                s = random.random()
                if s >= INT_EXPLORE_RATE:
                    out, hidden_int = self.int_args_net(input_int, hidden_int)
                    int_probs = F.softmax(out.view(-1), dim=0)
                    int_probs = int_probs.detach().cpu().numpy()
                    chosen_int = np.random.choice(len(self.int_values)+1, p=int_probs)
                    input_int = torch.zeros((1, len(self.int_values)+1)).to(device)
                    input_int[0, chosen_int] = 1
                    int_args.append(chosen_int)
                else:
                    chosen_int = None

                if t == SolType.IntTy:
                    arguments.append(self._select_int(contract, method, arg.evm_type.size, obs, chosen_int))
                elif t == SolType.UintTy:
                    arguments.append(self._select_uint(contract, method, arg.evm_type.size, obs, chosen_int))
            elif t == SolType.BoolTy:
                arguments.append(self._select_bool())
            elif t == SolType.StringTy:
                arguments.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg = self._select_slice(contract, method, sender, arg.evm_type.elem, obs, rnn_out)
                arguments.append(arg)
            elif t == SolType.ArrayTy:
                arg = self._select_array(contract, method, sender, arg.evm_type.size, arg.evm_type.elem, obs, rnn_out)
                arguments.append(arg)
            elif t == SolType.AddressTy:
                out, hidden_addr = self.addr_args_net(input_addr, hidden_addr)
                addr_probs = F.softmax(out.view(-1)[:len(self.addresses)], dim=0)
                addr_probs = addr_probs.detach().cpu().numpy()
                chosen_addr = np.random.choice(len(self.addresses[:len(addr_probs)]), p=addr_probs)
                arguments.append(self._select_address(sender, chosen_addr))
                input_addr = torch.zeros((1, 10)).to(device)
                input_addr[0, chosen_addr] = 1
                addr_args.append(chosen_addr)
            elif t == SolType.FixedBytesTy:
                arguments.append(self._select_fixed_bytes(arg.evm_type.size, obs))
            elif t == SolType.BytesTy:
                arguments.append(self._select_bytes(obs))
            else:
                assert False, 'type {} not supported'.format(t)
        return arguments, addr_args, int_args

    def _select_int(self, contract, method, size, obs, chosen_int=None):
        if chosen_int is not None and chosen_int != len(self.int_values):
            value = self.int_values[chosen_int]
            value &= ((1 << size) - 1)
            if value & (1 << (size - 1)):
                value -= (1 << size)
            return value

        p = 1 << (size - 1)
        return random.randint(-p, p-1)

    def _select_uint(self, contract, method, size, obs, chosen_int=None):
        if chosen_int is not None and chosen_int != len(self.int_values):
            value = self.int_values[chosen_int]
            value &= ((1 << size) - 1)
            return value

        p = 1 << size
        return random.randint(0, p-1)

    def _select_address(self, sender, idx=None):
        if sender in self.account_manager.attacker_indices:
            if idx is None:
                return random.choice(self.addresses)
            else:
                return self.addresses[idx]
        else:
            if idx is None or self.addresses[idx] in self.account_manager.attacker_addresses:
                l = [addr for addr in self.addresses if addr not in self.account_manager.attacker_addresses]
                return random.choice(l)
            else:
                return self.addresses[idx]

    def _select_bool(self):
        return random.choice([True, False])

    def _select_string(self, obs):
        bs = []
        size = random.randint(0, 40)
        for _ in range(size):
            bs.append(random.randint(1, 127))
        return bytearray(bs).decode('ascii')

    def _select_slice(self, contract, method, sender, typ, obs, rnn_out):
        if self.slice_size is None:
            size = random.randint(1, 15)
        else:
            size = self.slice_size
        return self._select_array(contract, method, sender, size, typ, obs, rnn_out)

    def _select_array(self, contract, method, sender, size, typ, obs, rnn_out):
        hidden_addr = rnn_out[0, 0].view(1, -1)
        input_addr = torch.zeros((1, 10)).to(device)
        hidden_int = rnn_out[0, 0].view(1, -1)
        input_int = torch.zeros((1, len(self.int_values)+1)).to(device)
        t = typ.t
        arr = []

        for _ in range(size):
            if t in (SolType.IntTy, SolType.UintTy):
                s = random.random()
                if s >= INT_EXPLORE_RATE:
                    out, hidden_int = self.int_args_net(input_int, hidden_int)
                    int_probs = F.softmax(out.view(-1), dim=0)
                    int_probs = int_probs.detach().cpu().numpy()
                    chosen_int = np.random.choice(len(self.int_values)+1, p=int_probs)
                    input_int = torch.zeros((1, len(self.int_values)+1)).to(device)
                    input_int[0, chosen_int] = 1
                else:
                    chosen_int = None

                if t == SolType.IntTy:
                    arr.append(self._select_int(contract, method, typ.size, obs, chosen_int))
                elif t == SolType.UintTy:
                    arr.append(self._select_uint(contract, method, typ.size, obs, chosen_int))
            elif t == SolType.BoolTy:
                arr.append(self._select_bool())
            elif t == SolType.StringTy:
                arr.append(self._select_string(obs))
            elif t == SolType.SliceTy:
                arg = self._select_slice(contract, method, sender, typ.elem, obs, rnn_out)
                arr.append(arg)
            elif t == SolType.ArrayTy:
                arg = self._select_array(contract, method, sender, typ.size, typ.elem, obs, rnn_out)
                arr.append(arg)
            elif t == SolType.AddressTy:
                out, hidden_addr = self.addr_args_net(input_addr, hidden_addr)
                addr_probs = F.softmax(out.view(-1)[:len(self.addresses)], dim=0)
                addr_probs = addr_probs.detach().cpu().numpy()
                chosen_addr = np.random.choice(len(self.addresses[:len(addr_probs)]), p=addr_probs)
                input_addr = torch.zeros((1, 10)).to(device)
                input_addr[0, chosen_addr] = 1
                arr.append(self._select_address(sender, chosen_addr))
            elif t == SolType.FixedBytesTy:
                arr.append(self._select_fixed_bytes(typ.size, obs))
            elif t == SolType.BytesTy:
                arr.append(self._select_bytes(obs))
            else:
                assert False, 'type {} not supported'.format(t)

        return arr

    def _select_fixed_bytes(self, size, obs):
        bs = []
        for _ in range(size):
            bs.append(random.randint(0, 255))
        return bs

    def _select_bytes(self, obs):
        size = random.randint(1, 15)
        return self._select_fixed_bytes(size, obs)

