from os import PRIO_PGRP
from torch import nn
import torch
import torch.nn.functional as F
from helper import *
# from torch_geometric.utils import softmax
from model.message_passing import scatter_


class GATLayer(torch.nn.Module):
   

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.W = get_param((4, 2 * out_features + nrela_dim, out_features))
        self.a = get_param((4, out_features, 1))
        self.x_linear = nn.Linear(in_features, out_features)

        # self.IterW = get_param((4 * out_features, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.layer_norm = nn.LayerNorm(out_features)



    # def forward(self, edge_index, x, edge_type_embed, row_ptr, sorted_idx):
    def forward(self, edge_index, x, edge_type_embed, ratio=None):
        
        if not isinstance(ratio, float):
            x_current = self.x_linear(x)
            if ratio == 'linear':
                ratio_schedule = ratio_linear_schedule(4, start=1.0, end=0.4)
            else:
                ratio_schedule = ratio_exponentia_schedule(4)
            # hiddens = []
            for i in range(1, 5):
                ratio = next(ratio_schedule)
                hidden_current = self.one_iter_aggregation(edge_index, x_current, edge_type_embed, i, ratio)
                x_current = hidden_current + x_current
                # hiddens.append(x_current)
            # x_current = torch.matmul(torch.cat(hiddens, dim=1), self.IterW)
        else:
            x_current = self.x_linear(x)
            for i in range(1, 5):
                x_current = self.one_iter_aggregation(edge_index, x_current, edge_type_embed, i, ratio)
        if self.concat:
            # if this layer is not last layer,
            x_current =  F.elu(x_current)

        return self.layer_norm(x_current)


    def one_iter_aggregation(self, edge_index, x, edge_type_embed, iter_layer,ratio):
        edge_h = torch.cat(
            (x[edge_index[0, :], :], x[edge_index[1, :], :], edge_type_embed[:, :]), dim=1)
       

        edge_h =  torch.matmul(edge_h, self.W[iter_layer - 1, :, :])
        # print('edge_h: ', edge_h.size())

        alpha = self.leakyrelu(torch.matmul(edge_h, self.a[iter_layer - 1, :, :]).squeeze())

        # print('alpha: ', alpha.size())
        alpha = softmax(alpha, edge_index[0], x.size(0))
        _, idx = alpha.topk(int(edge_type_embed.size(0) * ratio), dim=0)
        alpha_masked = torch.zeros_like(alpha, dtype=torch.bool)
        alpha_masked[idx] = True


        # set alpha=0 for not keep
        alpha = torch.where(alpha_masked, alpha, torch.zeros_like(alpha))

        out = self.path_message(edge_h, edge_index, size=x.size(0), edge_norm=alpha)
        
        return out


    def path_message(self, x, edge_index, size, edge_norm):
        # print('x bef: ', edge_norm)
        # print('x bef: ', x)
        

        x = edge_norm.unsqueeze(1) * x

        # print('x aff: ', x)
        out = scatter_('add', x, edge_index[0], dim_size=size)
        # print('out', out.size())
        return out

class GAT(torch.nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads, embed_dim):
        super(GAT, self).__init__()

        self.dropout = dropout
        nhid = embed_dim//nheads
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [GATLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        
        # self.W = get_param((relation_dim,  relation_dim))
        self.W = get_param((relation_dim,  embed_dim))

        self.out_att = GATLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, 
                                             embed_dim,
                                            #  relation_dim,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )
    
    def forward(self, edge_index, x, r, edge_type, edge_type_nhop, ratio=None):

        edge_embed = r[edge_type]
        edge_embed_nhop = r[edge_type_nhop[:, 0]] + r[edge_type_nhop[:, 1]]
        edge_embed = torch.cat((edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        # row_ptr, sorted_edge_index, sorted_idx = build_csr_from_edge_index(edge_index, x.size(0), device=x.device)

        # edge_index = sorted_edge_index
        # edge_embed = edge_embed[sorted_idx]

        # x = torch.cat([att( edge_index, x, edge_embed, row_ptr, sorted_idx) for att in self.attentions], dim=1)
        x = torch.cat([att(edge_index, x, edge_embed, ratio) for att in self.attentions], dim=1)

        x = self.dropout_layer(x)

        r = torch.matmul(r, self.W)


        edge_embed = r[edge_type]
        edge_embed_nhop = r[edge_type_nhop[:, 0]] + r[edge_type_nhop[:, 1]]
        edge_embed = torch.cat((edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        # edge_embed = edge_embed[sorted_idx]

        # x = F.elu(self.out_att(edge_index, x, edge_embed, row_ptr, sorted_idx))
        # x = F.elu(self.out_att(edge_index, x, edge_embed, ratio))
        x = self.out_att(edge_index, x, edge_embed, ratio)
        
        return x, r



class KBGAT_Model(torch.nn.Module):
    def __init__(self, edge_index, edge_type, params, feature_embeddings, indices_2hop):
        super(KBGAT_Model, self).__init__()

        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        num_rel = self.p.num_rel
        self.num_nodes = self.p.num_ent
        
        self.indices_2hop = indices_2hop

        if feature_embeddings != None:

            initial_entity_emb = feature_embeddings['entity_embedding']
            initial_relation_emb =  feature_embeddings['relation_embedding']

            self.init_embed = nn.Parameter(initial_entity_emb)
            self.init_rel = nn.Parameter(initial_relation_emb)

            self.entity_in_dim = initial_entity_emb.shape[1]
            self.relation_dim = initial_relation_emb.shape[1]
        else:
            self.init_embed		= get_param((self.num_nodes,   self.p.init_dim))
            self.init_rel = get_param(( num_rel*2, self.p.init_dim))
            self.entity_in_dim, self.relation_dim = self.p.init_dim, self.p.init_dim

       
        self.drop = self.p.hid_drop
        self.alpha = self.p.alpha
        self.nheads  =  self.p.nheads
        self.embed_dim = self.p.embed_dim
        self.dropout_layer = nn.Dropout(self.drop)
        # self.class_num = self.p.class_num

        # self.class_embedding = torch.nn.Embedding((self.class_num, self.class_num))
        # self.h_2_class_linear = torch.nn.Linear(self.embed_dim, self.class_num)
        # self.t_2_class_linear = torch.nn.Linear(self.embed_dim, self.class_num)

        # self.query = get_param((self.class_num, self.class_num))
        # self.key = get_param((self.class_num, self.class_num))
        # self.value = get_param((self.class_num, self.class_num))

        self.gat = GAT(self.num_nodes, self.entity_in_dim, self.p.gcn_dim, self.relation_dim,
                                    self.drop, self.alpha, self.nheads, self.embed_dim)

        self.W_entities  = get_param((self.entity_in_dim, self.embed_dim))
        self.bn = torch.nn.BatchNorm1d( self.embed_dim)

    def forward(self, feature=None):
        
        
        self.indices_2hop = self.indices_2hop.to(self.init_rel.device)
        
        self.edge_index = self.edge_index.to(self.init_rel.device)
        self.edge_type = self.edge_type.to(self.init_rel.device)

        edge_list_nhop = torch.cat(
            (self.indices_2hop[:, 3].unsqueeze(-1), self.indices_2hop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [self.indices_2hop[:, 1].unsqueeze(-1), self.indices_2hop[:, 2].unsqueeze(-1)], dim=1)

        
        self.edge_index = self.edge_index[[1,0]]

 
        edge_index = torch.cat((self.edge_index[:, :], edge_list_nhop[:, :]), dim=1)
        
        x, r = self.gat(edge_index, self.init_embed, self.init_rel, self.edge_type, edge_type_nhop, self.p.ratio)

        x_self = torch.matmul(self.init_embed, self.W_entities)
        x = x + x_self

        x = self.bn(x)
        x = self.dropout_layer(x)
        
        
        return x, r

    # def get_entities_class_attention(self, entities_embedding):
    #     score = torch.matmul(entities_embedding @ self.query, (self.class_embedding @ self.key).t())

    #     score = torch.softmax(score, dim=-1)

    #     weight_class_embeddings = torch.matmul(score, self.class_embedding.weight @ self.value)

    #     return weight_class_embeddings

