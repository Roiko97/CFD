import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from evaluation import *
import gaussian_diffusion as gd
from DNN import DNN
import scipy.sparse as sp
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl
from torch.optim import Adam, AdamW
import networkx as nx
import sys
DID = 0
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed(826)
torch.cuda.manual_seed_all(826)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CFD(nn.Module):
    def __init__(self, **kwargs):
        super(CFD, self).__init__()
        clusters = kwargs['clusters']
        lap = kwargs['lap']
        node2vec_path = kwargs['node2vec_path']
        mean_type = gd.ModelMeanType.START_X
        noise_schedule = 'linear-var'
        noise_scale = kwargs['noise_scale']
        noise_min = kwargs['noise_min']
        noise_max = kwargs['noise_max']
        steps = kwargs['steps']
        K = kwargs['K']
        time_dim = kwargs['time_dim']
        dropout = kwargs['dropout']
        self.lamda = kwargs['lamda']

        self.cluster_layer = Variable((torch.zeros(clusters, 16) + 1.).to(device), requires_grad=True)
        self.kmeans = KMeans(n_clusters=clusters, n_init=20)
        init_feat = self.get_feature(node2vec_path)
        self.self_z_0 = Variable(torch.from_numpy(init_feat).type(torch.float32).to(device), requires_grad=False)
        _ = self.kmeans.fit_predict(init_feat)
        self.cluster_layer.data = torch.tensor(self.kmeans.cluster_centers_.astype(np.float32)).to(device)
        self.v = 1.0

        self.diffusion = gd.GaussianDiffusion(mean_type, noise_schedule, noise_scale, noise_min, noise_max, steps, device).to(device)
        out_dims = kwargs["out_dims"]
        in_dims = out_dims[::-1]
        self.dnn = DNN(in_dims, out_dims, time_dim, community_dim=clusters, time_type="cat", norm=False, act_func="tanh", dropout = dropout).to(device)

        if kwargs["use_spricy"] is True:
            self.T_k = self.chebyshev_polynomials_sp(sp.csr_matrix(lap), K)
        else:
            self.T_k = self.chebyshev_polynomials(torch.from_numpy(lap.A).float().to(device), K)

    def chebyshev_polynomials_sp(self, L_tilde, K):
        N = L_tilde.shape[0]
        T_k = []
        T_k.append(sp.identity(N, format='csr'))
        if K >= 1:
            T_k.append(L_tilde)  # T_1
        for k in range(2, K + 1):
            T_k_k = 2 * L_tilde @ T_k[k - 1] - T_k[k - 2]
            T_k.append(T_k_k)
        return T_k

    def chebyshev_polynomials(self, L_tilde, K):
        N = L_tilde.shape[0]
        T_k = []
        T_k.append(torch.eye(N).to(device))  # T_0
        if K >= 1:
            T_k.append(L_tilde.to(device))  # T_1
        for k in range(2, K + 1):
            T_k_k = 2 * L_tilde @ T_k[k - 1] - T_k[k - 2]
            T_k.append(T_k_k.to(device))
        return T_k

    # TAG
    def graph_wavelet_filter(self, Z, T_k, theta):
        filtered_Z = torch.zeros_like(Z).to(device)
        for k in range(len(theta)):
            filtered_Z += theta[k] * (T_k[k]@ Z)
        return filtered_Z

    def batch_graph_wavelet_filter(self, Z, theta, idx):
        idx = idx.cpu()
        filtered_Z = torch.zeros_like(Z).to(device)
        for k in range(len(theta)):
            Tk = torch.from_numpy(self.T_k[k][idx][:, idx].A).to(torch.float32).to(device)
            filtered_Z += theta[k] * (Tk @ Z)
        return filtered_Z

    def assignment(self, cluster_centers, alpha=1):
        norm_squared = torch.sum((self.self_z_0.unsqueeze(1) - cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

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

if __name__ == "__main__":
    dataname = "cora"
    def load_data(name):
        r = "D:\\PyCharm_WORK\\MyCode\\Jung\\datasets\\attributed_nets\\"
        with open(r + name + ".pkl", 'rb') as f:
            data = pkl.load(f)
        return data['adj'], data['feat'], data['label']

    adj, feat, label = load_data(dataname)
    L = sp.csr_matrix(nx.normalized_laplacian_matrix(nx.from_numpy_array(adj.A)))
    clusters = len(np.unique(label))
    configs = {
        "clusters": clusters, # cluster number
        "lap": L, # Laplication matrix
        "node2vec_path": "TAG_embedding/%s_feature.emb" % dataname,
        "noise_scale": 0.1,
        "noise_min": 0.0001,
        "noise_max": 0.02,
        "steps": 100,
        "K": 2, #
        "time_dim": 10, # timestamp dim
        "dropout": 0.,
        "lamda": 0.001,
        "use_spricy": False,
        "out_dims": [32, 16]
    }
    clusDDM = CFD(**configs)

    opt = AdamW(clusDDM.parameters(), lr=0.01)

    for epoch in range(200):
        opt.zero_grad()
        clus_dis = clusDDM.assignment(clusDDM.cluster_layer)
        diff_loss = clusDDM.diffusion.training_losses(clusDDM.dnn, clusDDM.self_z_0, clus_dis, reweight=True)
        z = diff_loss['pred_xstart']
        loss = diff_loss["loss"].mean()

        loss.backward()
        opt.step()

        with torch.no_grad():
            sampling_noise = False
            sampling_step = 10
            pred_emb = clusDDM.diffusion.p_sample(clusDDM.dnn, z, clusDDM.assignment(clusDDM.cluster_layer), sampling_step, sampling_noise)
            acc, nmi, ari, f1 = eva(clusters, label, pred_emb)
            sys.stdout.write(
                'epoch = %d | Diffusion > ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f) \n' % (
                epoch, acc, nmi, ari, f1))
            sys.stdout.flush()

