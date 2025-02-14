from os import PRIO_PGRP
from torch import nn
import torch
import torch.nn.functional as F
from helper import *
from model.message_passing import MessagePassing
# from torch_geometric.utils import softmax
from torch_scatter import scatter_max, scatter_add, scatter_mean
from model.message_passing import scatter_

def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    # num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out

class CGATLayer(torch.nn.Module):
   

    def __init__(self, batch_size, num_nodes, num_relations,in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(CGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.relation_linear = nn.Linear(in_features, nrela_dim * num_relations)
        self.layer_linear = nn.Linear(in_features * 2, out_features)
        self.trans_readout_linear = nn.Linear(in_features, out_features)

        self.W = get_param((batch_size * in_features, out_features))
        self.a = get_param((4, out_features, 1))
        # self.linear = nn.Linear(in_features * 2, out_features)

        # self.IterW = get_param((4 , out_features, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = nn.LayerNorm(out_features)

    # def forward(self, edge_index, x, edge_type_embed, row_ptr, sorted_idx):
    def forward(self, edge_index, r_index, boudnary_input, query_input, ratio=None):
        
        if ratio is None:
            x_current = boudnary_input
            iter_num = 4
            ratio_start = 1.0
            ratio_end = 0.1
            # hiddens = []
            for i in range(1, iter_num + 1):
                ratio = ratio_start - (i- 1) * (ratio_start - ratio_end) / (iter_num - 1)
                ratio = max(ratio, ratio_end)
                # ratio = 1
                hidden_current = self.one_iter_aggregation(edge_index, r_index, x_current, query_input, i, ratio)
                x_current = hidden_current + x_current
                # hiddens.append(x_current)
            # x_current = torch.matmul(torch.cat(hiddens, dim=1), self.IterW)
        else:
            item = 4
            x_current = boudnary_input
            for i in range(item):
                h_current = self.one_iter_aggregation(edge_index, r_index, x_current, query_input, i, ratio)
                x_current = h_current + x_current

        return x_current
        # if self.concat:
        #     # if this layer is not last layer,
        #     return F.elu(x_current)
        # else:
        #     # if this layer is last layer,
        #     return self.layer_norm(x_current)


    def one_iter_aggregation(self, edge_index, r_index, boudnary_input, query, iter_layer, ratio):
        # E * B * ID
        node_input = boudnary_input[edge_index[0, :]]
        # B * ID -> R * B * ID
        relation_input = self.relation_linear(query).view(self.num_relations, query.size(0), self.nrela_dim)

        # R * B * ID -> E * B * ID
        edge_input = relation_input[r_index]    

        # E * B * ID -> E * B * ID
        message = node_input + edge_input

        # E * B * ID -> E * OD
        edge_h =  torch.matmul(message.view(message.size(0), -1), self.W[:message.size(1) * message.size(2),:])
        # print('edge_h: ', edge_h.size())
        # E * OD -> E * 1
        alpha = self.leakyrelu(torch.matmul(edge_h, self.a[iter_layer, :, :].suqeeze(0)).squeeze())

        alpha = softmax(alpha, edge_index[0], boudnary_input.size(0))
        _, idx = alpha.topk(int(edge_input.size(0) * ratio), dim=0)
        alpha_masked = torch.zeros_like(alpha, dtype=torch.bool)
        alpha_masked[idx] = True


        # set alpha=0 for not keep
        alpha = torch.where(alpha_masked, alpha, torch.zeros_like(alpha))

        # E * B * ID -> E * 2ID
        message = torch.cat([message, boudnary_input])

        update = self.path_message(message, edge_index, size=boudnary_input.size(0), edge_norm=alpha)

        readout = torch.mean(boudnary_input, dim=0)

        out = self.layer_linear(torch.cat([boudnary_input, update], dim=-1))
        out = out + self.trans_readout_linear(readout)

        out = self.layer_norm(out)
        out = F.elu(out)
        return out


    def path_message(self, x, edge_index, size, edge_norm):
        edge_norm = torch.concat([edge_norm, torch.ones(size, device=edge_index.device)], dim=0)
        x = edge_norm.unsqueeze(1).unsqueeze(-1) * x

        node_out_index = edge_index[1, :]
        node_out_index = torch.cat([node_out_index, torch.arange(size, device=edge_index.device)], dim=0)
        out = scatter_('add', x, node_out_index, dim_size=size)
        # print('out', out.size())
        return out

class CGAT(torch.nn.Module):
    def __init__(self, batch_size, num_nodes, num_relations, nfeat, nhid, relation_dim, dropout, alpha, nheads, embed_dim):
        super(CGAT, self).__init__()

        self.dropout = dropout
        nhid = embed_dim//nheads
        self.init_dim = nfeat
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [CGATLayer(self.batch_size, num_nodes, num_relations, nfeat,
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
        # self.W = get_param((relation_dim,  embed_dim))

        # self.out_att = CGATLayer(self.batch_size, num_nodes, num_relations,
        #                                      nhid * nheads if nheads > 1 else nfeat,
        #                                      embed_dim, 
        #                                      relation_dim,
        #                                     #  relation_dim,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False
        #                                      )
    
    def forward(self, edge_index, r_index, boudnary_input, query_input, ratio=None):


        # x = torch.cat([att( edge_index, x, edge_embed, row_ptr, sorted_idx) for att in self.attentions], dim=1)
        output = torch.cat([att( edge_index, r_index, boudnary_input, query_input, ratio) for att in self.attentions], dim=-1)

        # output = self.dropout_layer(output)

        # output = F.elu(self.out_att(edge_index, r_index, boudnary_input, output, ratio))

        node_query = query_input.expand(self.num_nodes, -1, -1)

        # N * B * 2OD
        output = torch.concat([output, node_query], dim=-1)

        return output



class CKBGAT_Model(torch.nn.Module):
    def __init__(self, edge_index, edge_type, params, feature_embeddings, indices_2hop):
        super(CKBGAT_Model, self).__init__()

        self.p = params
        self.edge_index = edge_index
        self.edge_type = edge_type
        num_rel = self.p.num_rel
        self.num_nodes = self.p.num_ent
        self.initialization = self.p.initialization_boudnary
        
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

        self.gat = CGAT(self.p.batch_size, self.num_nodes, num_rel * 2, self.entity_in_dim, self.p.gcn_dim, self.relation_dim,
                                    self.drop, self.alpha, self.nheads, self.embed_dim)

        self.W_entities  = get_param((self.entity_in_dim, self.embed_dim))
        self.bn = torch.nn.BatchNorm1d( self.embed_dim)

        self.logits_mlp = nn.Sequential(
            nn.Linear(self.embed_dim  + self.entity_in_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
    def bellmanford(self, h_index, r_index):
        device = h_index.device
        query = self.init_rel[r_index].to(device)
        # N * B * D
        boudnary = torch.rand(self.num_nodes, *query.shape, device=device)
        index = h_index.unsqueeze(-1).expand_as(query).to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)


        if self.initialization == "Zero-One":
            one = torch.ones(*query.shape).to(device)
            boudnary.scatter_add_(0, index.unsqueeze(0), one.unsqueeze(0))
        elif self.initialization == "Query":
            boudnary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        elif self.initialization == "QueryWithNoise":
            noise = torch.randn(*query.shape).to(device)
            boudnary.scatter_add_(
                0, index.unsqueeze(0), (torch.add(query, noise)).unsqueeze(0)
            )
        elif self.initialization == "AllZero":
            pass
        else:
            raise NotImplementedError

        outputs = self.gat(self.edge_index, self.edge_type, boudnary.to(device), query, self.p.ratio)

        return outputs

    def forward(self, h_index, r_index, t_index, feature=None):
        outputs = self.bellmanford(h_index, r_index)
        # inverse_outputs = self.bellmanford(t_index, r_index)
        # outputs = (outputs + inverse_outputs) / 2

        logits = self.logits_mlp(outputs).squeeze(-1).T
        
        return logits, None
    
    def get_loss(self, logits, objs):
        labels = torch.zeros_like(logits)
        labels[:, objs] = 1
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        pred = torch.sigmoid(logits)
        return pred, loss