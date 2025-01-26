import warnings
warnings.filterwarnings("ignore")

import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from utils import normalize_adj
from evaluation import *
import sys
import networkx as nx
import scipy.sparse as sp
from torch.optim import Adam
DID = 0
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed_all(826)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = self.glorot_init(input_dim, output_dim)
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class VGAE(nn.Module):
    def __init__(self, graph, hid1_dim, hid2_dim):
        super(VGAE, self).__init__()
        self.graph = graph
        self.label = graph.ndata['label']
        self.clusters = len(torch.unique(self.label))
        self.feat = graph.ndata['feat'].to(torch.float32)
        self.feat_dim = self.feat.shape[1]

        self.hid2_dim = hid2_dim
        self.hid1_dim = hid1_dim

        self.base_gcn = GraphConvSparse(self.feat_dim, hid1_dim)
        self.gcn_mean = GraphConvSparse(hid1_dim, hid2_dim, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hid1_dim, hid2_dim, activation = lambda x:x)


        self.adj = graph.adjacency_matrix().to_dense()
        G = nx.from_numpy_array(self.adj.numpy())
        self.L = sp.csr_matrix(nx.normalized_laplacian_matrix(G))


        self.adj = self.adj + torch.eye(self.graph.num_nodes())
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)
        self.pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

        self.adj_1 = torch.from_numpy(normalize_adj(self.adj.numpy()).A)



    def get_feature(self, feature_path):
        node_emb = dict()
        with open(feature_path, 'r') as reader:
            reader.readline()
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                node_id = embeds[0]
                node_emb[node_id] = embeds[1:]
            reader.close()
        feature = []
        for i in range(len(node_emb)):
            feature.append(node_emb[i])

        return np.array(feature)

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def forward(self):
        hidden = self.base_gcn(self.feat, self.adj_1)
        self.mean = self.gcn_mean(hidden, self.adj_1)
        self.logstd = self.gcn_logstddev(hidden, self.adj_1)
        gaussian_noise = torch.randn(self.feat.size(0), self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        A_pred = self.dot_product_decode(sampled_z)
        return sampled_z, A_pred


def load_data(name):
    with open(name + ".pkl", 'rb') as f:
        data = pkl.load(f)
    graph = dgl.from_scipy(data['adj'])
    graph.ndata['feat'] = torch.from_numpy(data['feat'].todense())
    graph.ndata['label'] = torch.from_numpy(data['label'])
    return graph


if __name__ == "__main__":
    name = "cora"
    graph = load_data(name)
    model = VGAE(graph, 32, 16)


    opt1 = Adam(model.parameters(), lr=0.01)

    weight_mask = model.adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = model.pos_weight

    max_nmi = 0
    max_acc = 0
    max_ari = 0
    max_f1 = 0
    for epoch in range(100):
        opt1.zero_grad()
        model.train()
        z, pred = model()
        loss = log_lik = model.norm * F.binary_cross_entropy(pred.view(-1), (model.adj).view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()

        loss -= kl_divergence
        loss.backward()
        opt1.step()

        with torch.no_grad():
            acc_or, nmi_or, ari_or, f1_or = eva(model.clusters, model.label.numpy(), z)
            sys.stdout.write(
                'VGAE >  ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc_or, nmi_or, ari_or, f1_or))
            sys.stdout.flush()

            if nmi_or > max_nmi:
                max_nmi = nmi_or
                max_acc = acc_or
                max_ari = ari_or
                max_f1 = f1_or

    sys.stdout.write(
        'VGAE best >  ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (max_acc, max_nmi, max_ari, max_f1))
    sys.stdout.flush()