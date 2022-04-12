import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ops import GCN, GraphUnet, Initializer, norm_g

class GAE(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(GAE, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.vec = GCN(args.l_dim, 1, self.n_act, 0.0)
        self.dec = GCN(1, args.l_dim, self.n_act, 0.0)
        self.out_l_1 = nn.Linear(args.l_dim, args.h_dim)
        self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        self.out_drop = nn.Dropout(p=args.drop_c)
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 100), requires_grad=True)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        hs,rs,ds = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels, rs, ds)

    def embed(self, gs, hs):
        o_hs = []
        o_rs = []
        o_ds = []
        for g, h in zip(gs, hs):
            h,r,d = self.embed_one(g, h)
            o_hs.append(h)
            o_rs.append(r)
            o_ds.append(d)
        
        hs = torch.stack(o_hs, 0)
        rs = torch.cat(o_rs, 0)
        ds = torch.cat(o_ds, 0)
        return hs,rs,ds

    def embed_one(self, g, h):
        # return to a vector
        g = norm_g(g)
        h = self.s_gcn(g, h)
        r = h
        v = torch.sigmoid(self.beta*self.vec(g, h))
        d = self.dec(g, v)
        h = torch.mm(h.t(),v).squeeze()
        return h,r,d

    def readout(self, hs):
        h_max = [torch.max(h, 0)[0] for h in hs]
        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)

    def metric(self, logits, labels, rs, ds):
        loss = F.nll_loss(logits, labels) + F.mse_loss(rs, ds)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc

class VGAE(nn.Module):
    def __init__(self, in_dim, n_classes, args):
        super(VGAE, self).__init__()
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)
        self.mu = GCN(args.l_dim, 1, self.n_act, 0.0)
        self.logvar = GCN(args.l_dim, 1, self.n_act, 0.0)
        self.dec = GCN(1, args.l_dim, self.n_act, 0.0)
        self.out_l_1 = nn.Linear(args.l_dim, args.h_dim)
        self.out_l_2 = nn.Linear(args.h_dim, n_classes)
        self.out_drop = nn.Dropout(p=args.drop_c)
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0, 100), requires_grad=True)
        Initializer.weights_init(self)

    def forward(self, gs, hs, labels):
        hs,rs,ds,mus,logvars = self.embed(gs, hs)
        logits = self.classify(hs)
        return self.metric(logits, labels, rs, ds, mus, logvars)

    def embed(self, gs, hs):
        o_hs = []
        o_rs = []
        o_ds = []
        o_mus = []
        o_logvars = []
        for g, h in zip(gs, hs):
            h,r,d,mu,logvar = self.embed_one(g, h)
            o_hs.append(h)
            o_rs.append(r)
            o_ds.append(d)
            o_mus.append(mu)
            o_logvars.append(logvar)
        
        hs = torch.stack(o_hs, 0)
        rs = torch.cat(o_rs, 0)
        ds = torch.cat(o_ds, 0)
        mus = torch.cat(o_mus, 0)
        logvars = torch.cat(o_logvars, 0)
        return hs,rs,ds,mus,logvars
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def embed_one(self, g, h):
        # return to a vector
        g = norm_g(g)
        h = self.s_gcn(g, h)
        r = h
        mu = self.mu(g, h)
        logvar = self.logvar(g, h)
        v = torch.sigmoid(self.beta*self.reparameterize(mu,logvar))
        d = self.dec(g, v)
        h = torch.mm(h.t(),v).squeeze()
        return h,r,d,mu,logvar

    def classify(self, h):
        h = self.out_drop(h)
        h = self.out_l_1(h)
        h = self.c_act(h)
        h = self.out_drop(h)
        h = self.out_l_2(h)
        return F.log_softmax(h, dim=1)
    
    def KL(self,mu,logvar):
        return -0.5 / mu.size(0) * torch.mean(torch.sum(
                1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    def metric(self, logits, labels, rs, ds, mus, logvars):
        loss = F.nll_loss(logits, labels) + self.KL(mus,logvars) + F.mse_loss(rs, ds)
        _, preds = torch.max(logits, 1)
        acc = torch.mean((preds == labels).float())
        return loss, acc