from helper import *

from torch import nn
import torch.nn.functional as F
from model.rgcn_conv import RGCNConv
from model.torch_rgcn_conv import FastRGCNConv

class RGCNModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params):
		super(RGCNModel, self).__init__()

		self.p = params
		self.edge_index = edge_index
		self.edge_type = edge_type
		self.act	= torch.tanh
		num_rel = self.p.num_rel

		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))

		self.init_rel = get_param(( num_rel*2, self.p.init_dim))
		self.w_rel 		= get_param((self.p.init_dim, self.p.embed_dim))

		self.input_linear = nn.Linear(self.p.init_dim, self.p.embed_dim)

		# self.drop = torch.nn.Dropout(self.p.hid_drop)
		

		# if self.p.gcn_layer == 1: self.act = None
		# self.rgcn_conv1 = RGCNConv(self.p.init_dim, self.p.gcn_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks, act=self.act, model_plus=self.p.model_plus)

		# self.rgcn_conv2 = RGCNConv(self.p.gcn_dim, self.p.embed_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks, ) if self.p.gcn_layer == 2 else None

		if self.p.gnn_distillation:
			if self.ratio_type == 'linear':
				ratio_schedule = ratio_linear_schedule(self.p.gcn_layer, start=1.0, end=0.4)
			elif self.p.ratio_type == 'exponential':
				ratio_schedule = ratio_exponentia_schedule(self.p.gcn_layer, start=1.0, gamma=0.74)
			else:
				raise ValueError('Invalid ratio type: {}'.format(self.p.ratio_type))

		self.rgcn_layers = nn.ModuleList()

		for i in range(self.p.gcn_layer):
			in_dim = self.p.embed_dim
			out_dim = self.p.embed_dim

            # 激活函数除了最后一层外，都使用
			act = self.act if i != self.p.gcn_layer - 1 else None

			if self.p.gnn_distillation:
				ratio = next(ratio_schedule)
			else:
				ratio = None

            # 创建 RGCN 层并添加到模块列表中
			self.rgcn_layers.append(
				RGCNConv(in_dim, out_dim, self.p.num_rel, self.p.rgcn_num_bases, self.p.rgcn_num_blocks, act=act, ratio=ratio, alpha=self.p.alpha)
			)
			# self.rgcn_layers.append(
			# 	nn.Dropout(self.p.hid_drop)
			# )

	def forward(self, feature=None):
		
		self.edge_index = self.edge_index.to(self.init_rel.device)
		self.edge_type = self.edge_type.to(self.init_rel.device)

		
		x = self.init_embed
		r = self.init_rel

		if feature != None:
			
			x = feature['entity_embedding']
			x = x.to(self.init_rel.device)
			
			if 'relation_embedding' in feature:
				r = Parameter(feature['relation_embedding'])
				r = r.to(self.init_rel.device)
				
		
		# x = self.rgcn_conv1(x, self.edge_index, self.edge_type)
		# x = self.drop(x)

		
		# x = self.rgcn_conv2(x, self.edge_index, self.edge_type) if self.p.gcn_layer == 2 else x
		# x = self.drop(x) if self.p.gcn_layer == 2 else x
		current_x = self.input_linear(x)
		for layer in self.rgcn_layers:
			if isinstance(layer, RGCNConv):
				hidden_x = layer(current_x, self.edge_index, self.edge_type)
			if self.p.gnn_distillation:
				current_x = hidden_x + current_x 
			else:
				if isinstance(layer, nn.Dropout):
					current_x = layer(hidden_x)
				else:
					current_x = hidden_x
			
		
		r = torch.matmul(r, self.w_rel)
		# print('rel weight:', self.w_rel)
		# print('rel weight norm', torch.norm(self.w_rel, p=2), (self.w_rel).mean(), self.w_rel.std())
		
		return current_x, r

		

