import torch
import torch.nn as nn
from domainadapt.models.layers_ker import GraphConvolution


class GCNDis(nn.Module):
    def __init__(self, feat, hid, par, emb_size, ker_size):
        super(GCNDis, self).__init__()

        self.gc1_1 = GraphConvolution(feat, hid, emb_size, ker_size)
        self.gc1_2 = nn.LeakyReLU(0.01)

        self.gc2_1 = GraphConvolution(feat + hid, par, emb_size, ker_size)
        self.gc2_2 = nn.LeakyReLU(0.01)

        self.gcn_poo = nn.AdaptiveAvgPool2d((1, par))

        self.lin1 = torch.nn.Linear(par, 32)
        self.lin1act = nn.LeakyReLU(0.01)

        self.lin2 = torch.nn.Linear(32, 16)
        self.lin2act = nn.LeakyReLU(0.01)

        self.lin3 = torch.nn.Linear(16, 1)
        self.lin3act = nn.Sigmoid()

    def forward(self, data):
        spec_domain = data.x[:, :3]

        x1 = self.gc1_1(data, spec_domain)
        x1 = self.gc1_2(x1)
        data.x = torch.cat((x1, data.x), 1)

        x2 = self.gc2_1(data, spec_domain)
        x2 = self.gc2_2(x2)

        x = self.gcn_poo(x2.unsqueeze(0))

        x = self.lin1(x.squeeze())
        x = self.lin1act(x)
        x = self.lin2(x)
        x = self.lin2act(x)
        x = self.lin3(x)
        x = self.lin3act(x)

        return x
