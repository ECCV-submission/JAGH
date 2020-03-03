import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.soft = nn.Softmax()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = F.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCN_img(nn.Module):
    def __init__(self, code_len, gamma):
        super(GCN_img, self).__init__()
        self.gc1 = GraphConvolution(4096, 2048)
        self.gc2 = GraphConvolution(2048, 2048)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2048, code_len)
        self.alpha = 1.0

        self.gamma = gamma

        self.weight = nn.Parameter(torch.FloatTensor(32, 32))
        nn.init.constant_(self.weight, 1e-6)

    def forward(self, hid, adj):

        adj = adj + self.gamma * self.weight

        feat = self.relu(self.gc1(hid, adj))
        feat = self.relu(self.gc2(feat, adj))
        feat = self.linear(feat)
        code = torch.tanh(self.alpha * feat)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCN_txt(nn.Module):
    def __init__(self, code_len, gamma):
        super(GCN_txt, self).__init__()
        self.gc1 = GraphConvolution(4096, 2048)
        self.gc2 = GraphConvolution(2048, 2048)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2048, code_len)
        self.alpha = 1.0

        self.gamma = gamma

        self.weight = nn.Parameter(torch.FloatTensor(32, 32))
        nn.init.constant_(self.weight, 1e-6)

    def forward(self, hid, adj):
        adj =  adj + self.gamma * self.weight
        feat = self.relu(self.gc1(hid, adj))
        feat = self.relu(self.gc2(feat, adj))
        feat = self.linear(feat)
        code = torch.tanh(self.alpha * feat)
        return code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)