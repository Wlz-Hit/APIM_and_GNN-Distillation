from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from model.compgcn_conv import CompGCNConv




		
class CompGCNBase(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params=None):
		super(CompGCNBase, self).__init__()

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p = params
		self.act	= torch.tanh
		num_rel = self.p.num_rel
		
		
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		# self.device		= self.edge_index.device

		if self.p.compgcn_num_bases > 0:
			self.init_rel  = get_param((self.p.compgcn_num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		self.cpmpgcn_layers = nn.ModuleList()
		if self.p.gnn_distillation:
			if self.ratio_type == 'linear':
				ratio_schedule = ratio_linear_schedule(self.p.gcn_layer, start=1.0, end=self.p.ratio)
			elif self.p.ratio_type == 'exponential':
				ratio_schedule = ratio_exponentia_schedule(self.p.gcn_layer, start=1.0, gamma=self.p.ratio)
			else:
				raise ValueError('Invalid ratio type: {}'.format(self.p.ratio_type))
				

		for i in range(self.p.gcn_layer):
			if self.p.gnn_distillation:
				ratio = next(ratio_schedule)
			else:
				ratio = None
			if self.p.compgcn_num_bases > 0:
				self.cpmpgcn_layers.append(CompGCNConvBasis(self.p.embed_dim if i > 0 else self.p.init_dim, self.p.embed_dim, num_rel, self.p.compgcn_num_bases, ratio=ratio, alpha=self.p.alpha, act=self.act, params=self.p))
			else:
				self.cpmpgcn_layers.append(CompGCNConv(self.p.embed_dim if i > 0 else self.p.init_dim, self.p.embed_dim, num_rel, ratio=ratio, alpha=self.p.alpha, act=self.act, params=self.p))

		# if self.p.compgcn_num_bases > 0:
		# 	for i in range(self.p.gcn_layer):
		# 		self.cpmpgcn_layers.append(CompGCNConvBasis(self.p.gcn_dim if i > 0 else self.p.init_dim, self.p.gcn_dim, num_rel, self.p.compgcn_num_bases, act=self.act, params=self.p))
			
		# 	# self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.compgcn_num_bases, act=self.act, params=self.p)
		# 	# self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		# else:
		# 	for i in range(self.p.gcn_layer):
		# 		self.cpmpgcn_layers.append(CompGCNConv(self.p.gcn_dim if i > 0 else self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p))
			# self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			# self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		
		# self.drop1 = torch.nn.Dropout(self.p.hid_drop)
		# self.drop2 = torch.nn.Dropout(self.p.feat_drop) if self.p.score_func.lower() == 'conve' else torch.nn.Dropout(self.p.hid_drop) 

		
	def forward(self, feature=None):

		self.edge_index = self.edge_index.to(self.init_rel.device)
		self.edge_type = self.edge_type.to(self.init_rel.device)

		

		# print('ini: ', self.init_rel)
			
		
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

		x = self.init_embed

		if feature != None:

			x = feature['entity_embedding']
			x = x.to(self.init_rel.device)
			if 'relation_embedding' in feature:
				r = Parameter(feature['relation_embedding'])
				r = r.to(self.init_rel.device)
			
			
		
		for layer in self.cpmpgcn_layers:
			x, r = layer(x, self.edge_index, self.edge_type, rel_embed=r)

		# x, r	= self.conv1(x, self.edge_index, self.edge_type, rel_embed=r)
		
		
		# x	= self.drop1(x)
		# x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		# x	= self.drop2(x) 							if self.p.gcn_layer == 2 else x
		

		# sub_emb	= torch.index_select(x, 0, sub)
		# rel_emb	= torch.index_select(r, 0, rel)
		return  x, r


