import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from .amounts import AMOUNTS
from .addr_map import ADDR_MAP


HIDDEN_PARAMS = 100


class ArgsNet(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ArgsNet, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.gru = nn.GRUCell(self.input_size, self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size, 50)
        self.fc2 = nn.Linear(50, self.input_size)

    def forward(self, input, hidden):
        new_hidden = self.gru(input, hidden)
        out = F.relu(self.fc1(new_hidden))
        out = self.fc2(out)
        return out, new_hidden

    
class ParamsNet(nn.Module):

    def __init__(self, input_size):
        super(ParamsNet, self).__init__()

        self.input_size = input_size

        self.fc1_addr = nn.Linear(self.input_size, HIDDEN_PARAMS)
        # self.bn1_addr = nn.BatchNorm1d(HIDDEN_PARAMS)
        self.fc2_addr = nn.Linear(HIDDEN_PARAMS, HIDDEN_PARAMS)
        # self.bn2_addr = nn.BatchNorm1d(HIDDEN_PARAMS)
        self.final_fc_addr = nn.Linear(HIDDEN_PARAMS, len(ADDR_MAP))

        self.fc1_amount = nn.Linear(self.input_size, HIDDEN_PARAMS)
        self.fc2_amount = nn.Linear(HIDDEN_PARAMS, HIDDEN_PARAMS)
        self.fc3_amount = nn.Linear(HIDDEN_PARAMS, len(AMOUNTS))
        

    def predict_sender(self, x):
        assert x.size()[1] == self.input_size
        
        x_addr = F.relu(self.fc1_addr(x))
        # x_addr = self.bn1_addr(x_addr)
        x_addr = F.relu(self.fc2_addr(x_addr))
        # x_addr = self.bn2_addr(x_addr)
        x_addr = self.final_fc_addr(x_addr)
        return x_addr

    def predict_amount(self, x):
        assert x.size()[1] == self.input_size
        
        x_addr = F.relu(self.fc1_amount(x))
        x_addr = F.relu(self.fc2_amount(x_addr))
        x_addr = self.fc3_amount(x_addr)
        return x_addr


class EmbedGCN(nn.Module):

    def __init__(self, n_feat, n_hid, n_embed):
        super(EmbedGCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, 5*n_hid)
        # self.bn1 = nn.BatchNorm1d(3*n_hid)

        self.gc2 = GraphConvolution(5*n_hid, 3*n_hid)
        # self.bn2 = nn.BatchNorm1d(n_hid)

        self.gc3 = GraphConvolution(3*n_hid, n_hid)
        # self.bn3 = nn.BatchNorm1d(n_hid)
        
        # self.gc4 = GraphConvolution(n_hid, n_hid)
        # self.bn4 = nn.BatchNorm1d(n_hid)
        
        # self.gc5 = GraphConvolution(n_hid, n_hid)
        # self.bn5 = nn.BatchNorm1d(n_hid)
        
        self.gc6 = GraphConvolution(n_hid, n_embed)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = self.bn1(x)
        
        x = F.relu(self.gc2(x, adj))
        # x = self.bn2(x)

        x = F.relu(self.gc3(x, adj))
        # x = self.bn3(x)

        # x = F.relu(self.gc4(x, adj))
        # x = self.bn4(x)

        # x = F.relu(self.gc5(x, adj))
        # x = self.bn5(x)
        
        x = self.gc6(x, adj)

        return x

class PolicyNet(nn.Module):

    def __init__(self, raw_method_size, method_size, state_size):
        super(PolicyNet, self).__init__()
        self.raw_method_size = raw_method_size
        self.method_size = method_size
        self.state_size = state_size
        
        self.fc1 = nn.Linear(self.state_size, 200)
        self.bn1 = nn.BatchNorm1d(200)
        
        self.fc3 = nn.Linear(2*self.method_size, 200)
        self.bn3 = nn.BatchNorm1d(200)
        
        self.fc = nn.Linear(400, 100)
        self.bn = nn.BatchNorm1d(100)
        
        self.fc_function = nn.Linear(100, 50)
        self.bn_function = nn.BatchNorm1d(50)

        self.fc_function2 = nn.Linear(50, 1)

        # Layers for compression of feature map
        
        self.fc_feat1 = nn.Linear(self.raw_method_size, 200)
        # self.bn_feat1 = nn.BatchNorm1d(200)

        self.fc_feat2 = nn.Linear(200, 100)
        # self.bn_feat2 = nn.BatchNorm1d(100)
        
        self.fc_feat3 = nn.Linear(100, self.method_size)

    def compress_features(self, x):
        x = F.relu(self.fc_feat1(x))
        x = F.relu(self.fc_feat2(x))
        x = torch.sigmoid(self.fc_feat3(x))
        return x

    def predict_method(self, x_state, x_method):
        assert x_state.size()[1] == self.state_size, '{} vs {}'.format(x_state.size()[1], self.state_size)
        assert x_method.size()[1] == 2*self.method_size, '{} vs {}'.format(x_method.size()[1], self.method_size)
        
        x_state = F.relu(self.fc1(x_state))
        # x_state = self.bn1(x_state)
        
        x_method = F.relu(self.fc3(x_method))
        # x_method = self.bn3(x_method)

        x = torch.cat([x_state, x_method], dim=1)
        x = F.relu(self.fc(x))
        # x = self.bn(x)

        x = F.relu(self.fc_function(x))
        # x = self.bn_function(x)

        return self.fc_function2(x)

