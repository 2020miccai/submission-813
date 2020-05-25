import torch
import torch.nn as nn
from domainadapt.models.layers_ker import GraphConvolution


class GCNSeg(nn.Module):
    def __init__(self, feat, hid, par, emb_size, ker_size):
        super(GCNSeg, self).__init__()

        self.gc1_1 = GraphConvolution(feat, hid, emb_size, ker_size)
        self.gc1_2 = nn.LeakyReLU(0.01)

        hid1 = torch.div(hid, 2).item()
        self.gc2_1 = GraphConvolution(feat + hid, hid1, emb_size, ker_size)
        self.gc2_2 = nn.LeakyReLU(0.01)

        self.gc3_1 = GraphConvolution(feat + hid + hid1, par, emb_size, ker_size)
        self.gc3_2 = nn.LeakyReLU(0.01)

    def forward(self, data):
        spec_domain = data.x[:, :3]

        x1 = self.gc1_1(data, spec_domain)
        x1 = self.gc1_2(x1)
        data.x = torch.cat((x1, data.x), 1)

        x2 = self.gc2_1(data, spec_domain)
        x2 = self.gc2_2(x2)
        data.x = torch.cat((x2, data.x), 1)

        x3 = self.gc3_1(data, spec_domain)
        x3 = self.gc3_2(x3)

        return x3
