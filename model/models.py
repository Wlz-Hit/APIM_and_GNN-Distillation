# from model.compgcn_model import CompGCN_TransE, CompGCN_DistMult, CompGCN_ConvE
from model.compgcn_model import CompGCNBase
from model.rgcn_model import RGCNModel
from model.KBGAT import KBGAT_Model
from model.CKBGAT import CKBGAT_Model
from helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp(torch.nn.Module):

	def __init__(self, in_channels, out_channels, act, params):
		super(mlp, self).__init__()

		self.p = params
		self.act	= act

		self.W_entity	= get_param((in_channels, out_channels))
		self.W_relation = get_param((in_channels, out_channels))
		self.bn	= torch.nn.BatchNorm1d(out_channels)

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		if self.p.bias:  self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, r):
		out =  torch.mm(x, self.W_entity)

		if self.p.bias: out = out + self.bias

		out = self.bn(out)
		if self.act is not None:
			out = self.act(out)


		return out, torch.matmul(r, self.W_relation)	


class BaseModel(torch.nn.Module):
	def __init__(self, edge_index, edge_type, params, feature_embeddings=None, indices_2hop=None):
		super(BaseModel, self).__init__()

		self.p		= params
		
		#### loss
		self.bceloss	= torch.nn.BCELoss()   ##bce loss
		self.logsoftmax_loss = torch.nn.CrossEntropyLoss(reduction='mean')
		self.margin = self.p.margin   ####for margin loss
		self.log_temp = torch.nn.Parameter(torch.tensor(1.0 / self.p.temp).log())


		self.edge_index = edge_index
		self.edge_type = edge_type
		self.act = torch.tanh
	
		
		self.inter_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		if self.p.model == 'random' and self.p.score_func.lower() == 'conve':
			if self.p.init_dim != self.p.embed_dim:
				self.p.init_dim = self.p.embed_dim

		if self.p.model == 'random' or self.p.model == 'mlp':
			self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
			if self.p.score_func == 'transe': 	self.init_rel = get_param((self.p.num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((self.p.num_rel*2, self.p.init_dim))

		if self.p.pretrain_epochs > 0:
			self.abstract_entity_embedding =  nn.Embedding(self.p.num_ent, self.p.embed_dim)
			self.abstract_relation_embedding =  nn.Embedding(self.p.num_rel * 2, self.p.embed_dim)
			nn.init.xavier_uniform_(self.abstract_entity_embedding.weight)
			nn.init.xavier_uniform_(self.abstract_relation_embedding.weight)
		
		# if self.p.score_func == 'transe': 	self.init_rel = get_param((self.p.num_rel,   self.p.init_dim))
		# else: 					self.init_rel = get_param((self.p.num_rel*2, self.p.init_dim))

		# self.init_rel = get_param((edge_type.unique().size(0), self.p.init_dim))

		if self.p.model == 'compgcn':
			
			self.model = CompGCNBase(self.edge_index, self.edge_type, self.p)
			

		elif self.p.model == 'rgcn':
			
			self.model = RGCNModel(self.edge_index, self.edge_type, self.p)
			# self.init_rel = get_param(( self.p.num_rel, self.p.init_dim))
		
		elif self.p.model == 'kbgat':
			
			self.model = KBGAT_Model(self.edge_index, self.edge_type, self.p, feature_embeddings, indices_2hop)
		else:
			self.model = CKBGAT_Model(self.edge_index, self.edge_type, self.p, feature_embeddings, indices_2hop)


		self.TransE_score = TransE_score(self.p)
		self.DistMult_score = DistMult_score(self.p)
		self.ConvE_score = ConvE_score(len(self.edge_type.unique()), self.p)
		self.Cosine_score = Cosine_score()
		# self.Clissificer_score = Classificer_score(self.p)
		self.Abstraction_score = AbstractFeature_score(self.p)

		self.mlp1 = mlp(self.p.init_dim, self.inter_dim, self.act, self.p)
		self.mlp2 = mlp(self.inter_dim, self.p.embed_dim, self.act, self.p) if self.p.gcn_layer == 2 else None

		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.invest = 1

		self.possibility_alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=self.p.loss_alpha_update)

	def loss(self, pred, true_label, original_score=None, pos_neg_ent=None):

		if self.p.loss_func == 'bce':
			if self.invest == 1:
				print('loss function: BCE')
			# original_score = torch.sigmoid(original_score)
			# loss = self.bceloss(original_score, true_label)
			# if self.p.model_plus:
			# 	pred = torch.sigmoid(pred)
			# 	loss = loss + self.loss_alpha * self.bceloss(pred, true_label)
			# 	original_score = original_score +  pred
			pred = torch.sigmoid(pred)
			loss = self.bceloss(pred, true_label)
		else:
			if self.invest == 1:
				# TODO Infonce loss
				print('loss function: INFONCE')
			loss = self.logsoftmax_loss(original_score, true_label)
			# if self.p.model_plus:
			# 	loss = loss + self.loss_alpha * self.logsoftmax_loss(pred, true_label)
			# 	original_score = original_score + self.loss_alpha * pred

		self.invest = 0
		
		return loss, pred

	def get_loss(self, x, r, sub, rel, obj ,label, pos_neg_ent):
	
		if self.p.model == 'ckbgat':
			self.invest = 0
			return self.model.get_loss(x, obj)
		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		if self.p.score_func.lower() == 'transe':
			if self.invest == 1:
				print('score function: transe')
				
			score, original_score = self.TransE_score(sub_emb, rel_emb, x, label, pos_neg_ent)
		
		elif self.p.score_func.lower() == 'distmult':
			if self.invest == 1:
				print('score function: distmult')

			score , original_score= self.DistMult_score(sub_emb, rel_emb, x, label, pos_neg_ent)
		
		elif self.p.score_func.lower() == 'conve':
			if self.invest == 1:
				print('score function: conve')
	
			score, original_score = self.ConvE_score(sub_emb, rel_emb, rel, x, label, pos_neg_ent)


		elif self.p.score_func.lower() == 'cosine':
			if self.invest == 1:
				print('score function: cosin')
			
			hr_class_embedding = self.model.hr_2_class_linear(sub_emb, dim=-1)
			tail_class_embedding = self.model.t_2_class_linear(x)
			hr_class_embedding = self.model.get_entities_class_attention(hr_class_embedding)
			tail_class_embedding = self.model.get_entities_class_attention(tail_class_embedding)
			score, original_score = self.Cosine_score(hr_class_embedding, tail_class_embedding)

		# 1、possibility score 计算方法 修改
		# 2、possibility L2 正则化项 修改
		# 3、possibility 初始化 修改
		# 4、function combin 计算方法修改
		# 4、possibility 注意力机制
		# 5、possibility 学习率调整
		# 6、possibility 多头
		if self.p.model_plus:
			if self.p.pretrain_epochs > 0:
				tucker_logits, possibility_logits = self.Abstraction_score(self.abstract_entity_embedding(sub), self.abstract_relation_embedding(rel), self.abstract_entity_embedding.weight, 
																label, logits=score, pos_neg_ent=pos_neg_ent)
			else:
				tucker_logits, possibility_logits = self.Abstraction_score(sub_emb, rel_emb, rel, x, label, logits=score, pos_neg_ent=pos_neg_ent)

			combin_logits = self.tucker_alpha * tucker_logits + self.possibility_alpha * possibility_logits
			# if not self.training:
			# 	_, combin_logits_mask = torch.topk(combin_logits, k=self.p.candidate_num, dim=1)
			# 	combin_logits_mask = torch.zeros_like(combin_logits).scatter_(1, combin_logits_mask, 1)

			# 	final_logits = combin_logits + score
			# 	final_logits = final_logits.masked_fill(~combin_logits_mask.bool(), 1e-9) 
			# else:
			final_logits = combin_logits + score

			original_loss, original_score = self.loss(original_score, label, score, pos_neg_ent)
			final_loss, final_score = self.loss(final_logits, label, score, pos_neg_ent)

			tucker_loss, tucker_score = self.loss(tucker_logits, label, score, pos_neg_ent)
			possibility_loss, possibility_score = self.loss(possibility_logits, label, score, pos_neg_ent)
			# loss, score = self.loss(score, label, original_score, pos_neg_ent)

			# loss = loss + self.tucker_alpha * possibility_loss + self.possibility_alpha * tucker_loss + 1e-4 * torch.norm(self.Abstraction_score.possibility_codebook.data, p=2)

			final_loss = final_loss + self.tucker_alpha * tucker_loss + self.possibility_alpha * possibility_loss + original_loss
			if self.p.candidate_num > 0:
				final_loss = final_loss + 1e-4 * torch.norm(self.Abstraction_score.core_tensor.data, p=2)
			final_loss = final_loss + 1e-4 * torch.norm(self.Abstraction_score.possibility_codebook.data, p=2)
			return (final_score, original_score , tucker_score, possibility_score), final_loss
		else:
			loss, score = self.loss(score, label, original_score, pos_neg_ent)
			return score, loss


	def forward(self, h_index, r_index, t_index, feature=None):

		if self.p.model == 'random':

			if self.invest == 1:
				
				print('investigation mode: random initialization')
				if feature != None:
					print('use feature as input!!!!')
			
			r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x 	= self.init_embed

			x	= self.drop(x)

		
		elif  self.p.model == 'mlp':
			if self.invest == 1:
				print('investigation mode: mlp')
				if feature != None:
					print('use feature as input!!!!')
		
			r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x = self.init_embed

			x, r = self.mlp1(x, r)
			x	= self.drop(x)

			x, r = self.mlp2(x, r) if self.p.gcn_layer == 2 else (x, r)
			x	= self.drop(x) if self.p.gcn_layer == 2 else x

		else:
			if self.invest == 1:
				print('investigation mode: aggregation')
			if self.model.p.model == 'ckbgat':
				x, r = self.model(h_index, r_index, t_index)
			else:
				x, r = self.model()

		return x, r

class Cosine_score(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
	
	def forward(self, hr_embedding, t_embedding):
		hr_embedding = torch.nn.functional.normalize(hr_embedding, dim=-1)
		t_embedding = torch.nn.functional.normalize(t_embedding, dim=-1)

		scores = hr_embedding @ t_embedding.t()

		return scores, 0

class AbstractFeature_score(torch.nn.Module):
	# P: finetune_t, candidate_num, infoENCE/Contrastive_learning
	def __init__(self, param=None) -> None:
		super().__init__()
		self.p = param

		self.possibility_codebook = torch.nn.Parameter(torch.Tensor(self.p.num_rel * 2, self.p.class_num, self.p.class_num))
		torch.nn.init.xavier_uniform_(self.possibility_codebook.data)
		if self.p.candidate_num > 0:
			self.core_tensor = get_param((self.p.class_num, self.p.class_num, self.p.class_num))
			self.bnn0 = torch.nn.BatchNorm1d(self.p.class_num)
			self.bnn1 = torch.nn.BatchNorm1d(self.p.class_num)
		
		# self.head_abstract_feature = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		# self.tail_abstract_feature = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		# self.relation_abstract_feature = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		# self.hr_abstract_feature = torch.nn.Linear(self.p.class_num * 2, self.p.class_num)

		self.head_sub_graph_feature =  nn.Sequential(
										nn.Linear(self.p.embed_dim, self.p.embed_dim),
										nn.ReLU(),
										nn.Linear(self.p.embed_dim, self.p.class_num),  # class_num => for T+P usage
										nn.Dropout(0.1)
									)

		self.relation_sub_graph_feature = nn.Sequential(
										nn.Linear(self.p.embed_dim, self.p.embed_dim),
										nn.ReLU(),
										nn.Linear(self.p.embed_dim, self.p.class_num),  # class_num => for T+P usage
										nn.Dropout(0.1)
									)
		self.tail_sub_graph_feature = nn.Sequential(
										nn.Linear(self.p.embed_dim, self.p.embed_dim),
										nn.ReLU(),
										nn.Linear(self.p.embed_dim, self.p.class_num),  # class_num => for T+P usage
										nn.Dropout(0.1)
										)
		
		self.hr_abstract_feature = nn.Sequential(
										nn.Linear(self.p.embed_dim * 2, self.p.class_num * 2),
										nn.ReLU(),
										nn.Linear(self.p.class_num * 2, self.p.class_num * 2),
										nn.ReLU(),
										nn.Linear(self.p.class_num * 2, self.p.class_num),
										nn.Dropout(0.1)
									)
		self.tail_abstract_feature = nn.Sequential(
										nn.Linear(self.p.embed_dim, self.p.embed_dim),
										nn.ReLU(),
										nn.Linear(self.p.embed_dim, self.p.class_num),  # class_num => for T+P usage
										nn.Dropout(0.1)
									)
		
		self.msg_fc = nn.Linear(self.p.class_num*2, self.p.class_num)
		self.agg_fc = nn.Linear(self.p.class_num, self.p.class_num)
		# self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / self.p.t).log(), requires_grad=self.p.finetune_t)

		# self.linear_score = torch.nn.Linear(self.p.embed_dim * self.p.topk, 1)
		# self.linear_score = torch.nn.Linear(self.p.class_num * 2, 1)

	def forward(self, head_vector, relation_vector, relation_index, tail_vector, label, logits=None, pos_neg_ent=None)->torch.tensor:

		if self.p.candidate_num > 0:
			head_sub_graph_feature = self.head_sub_graph_feature(head_vector)
			relation_sub_graph_feature = self.relation_sub_graph_feature(relation_vector)
			tail_sub_graph_feature = self.tail_sub_graph_feature(tail_vector)


			head_abstract_feature = self.bnn0(head_sub_graph_feature)
			head_abstract_feature = head_abstract_feature.view(-1, 1, self.p.class_num)

			W_mat = torch.mm(relation_sub_graph_feature, self.core_tensor.view(relation_sub_graph_feature.size(1), -1))
			W_mat = W_mat.view(-1, head_abstract_feature.size(-1), tail_sub_graph_feature.size(-1))

			# (B * 1 * K) * (1 * K * K) = (B * K * K) 
			W_mat = torch.bmm(head_abstract_feature, W_mat)
			W_mat = W_mat.view(-1, tail_sub_graph_feature.size(1))
			W_mat = self.bnn1(W_mat)
			# logits = torch.mm(W_mat, tail_abstract_feature.t()) * self.log_inv_t.exp()
			tucker_logits = torch.mm(W_mat, tail_sub_graph_feature.t())

			# if self.p.candidate_num > 0:
			# 	# _, logits_topk_index = torch.topk(possibility_score, self.p.candidate_num, dim=-1)
			# 	# _, logits_bottom_index = torch.topk(possibility_score, self.p.candidate_num, dim=-1, largest=False)
			# 	_, logits_topk_index = torch.topk(possibility_score, self.p.candidate_num, dim=-1)
			# 	_, logits_bottom_index = torch.topk(possibility_score, self.p.candidate_num, dim=-1, largest=False)
			# 	logits_index = torch.cat([logits_topk_index, logits_bottom_index], dim=-1)
			# 	logits_mask =  torch.zeros_like(tucker_logits).scatter_(-1, logits_index, 1)
				# tucker_logits = tucker_logits.masked_fill(~logits_mask.bool(), 1e-9)

			# 	if self.training:
			# 		label_mask = (label > 0.01).bool()
			# 		tucker_logits = tucker_logits.masked_fill(~logits_mask.bool() & ~label_mask.bool(), 1e-9)
			# 	else:
			# 		tucker_logits = tucker_logits.masked_fill(~logits_mask.bool(), 1e-9)
		else:
			tucker_logits = torch.zeros_like(logits)

		hr_abstract_feature = self.hr_abstract_feature(torch.cat([head_vector, relation_vector], dim=-1))
		tail_abstract_feature = self.tail_abstract_feature(tail_vector)

		pc = torch.tanh(self.possibility_codebook[relation_index,:,:])
		# pc = torch.sigmoid(self.possibility_codebook)


		if self.p.topk > 0:
			# _, hr_abstract_topk_index = torch.topk(hr_abstract_feature, self.p.topk, dim=-1)
			# _, tail_abstract_topk_index = torch.topk(tail_abstract_feature, self.p.topk, dim=-1)
			# hr_abstract_topk_mask = torch.zeros_like(hr_abstract_feature).scatter_(-1, hr_abstract_topk_index, 1)
			# tail_abstract_topk_mask = torch.zeros_like(tail_abstract_feature).scatter_(-1, tail_abstract_topk_index, 1)

			# # intermediate = torch.bmm(hr_abstract_topk_mask.unsqueeze(1), pc.expand(hr_abstract_topk_mask.size(0), self.p.class_num, self.p.class_num))  # batch_size x 1 x K

        	# # # 再与 tail_mask 进行乘法： batch_size x 1 x K * batch_size x K x t -> batch_size x 1 x t
			# # possibility_score = torch.bmm(intermediate, tail_abstract_topk_mask.unsqueeze(0).expand(hr_abstract_topk_mask.size(0), tail_abstract_topk_mask.size(0), tail_abstract_topk_mask.size(1)).transpose(1, 2)).squeeze(1)  # batch_size

			# hr_abstract_topk_feature = torch.where(hr_abstract_topk_mask.bool(), hr_abstract_feature, torch.zeros_like(hr_abstract_feature))
			# tail_abstract_topk_feature = torch.where(tail_abstract_topk_mask.bool(), tail_abstract_feature, torch.zeros_like(tail_abstract_feature))

			hr_mask_soft = soft_topk_mask(hr_abstract_feature, k=self.p.topk, temp=0.5)
			tail_mask_soft = soft_topk_mask(tail_abstract_feature, k=self.p.topk, temp=0.5)

			# multiply => keep partial
			hr_abstract_topk_feature = hr_mask_soft * hr_abstract_feature
			tail_abstract_topk_feature = tail_mask_soft * tail_abstract_feature

			# intermediate = torch.bmm(hr_abstract_topk_feature.unsqueeze(1), pc.expand(hr_abstract_topk_feature.size(0), self.p.class_num, self.p.class_num))  # batch_size x 1 x K
			intermediate = torch.bmm(hr_abstract_topk_feature.unsqueeze(1), pc)  # batch_size x 1 x K
			possibility_score = torch.bmm(intermediate, tail_abstract_topk_feature.unsqueeze(0).expand(hr_abstract_topk_feature.size(0), tail_abstract_topk_feature.size(0), tail_abstract_topk_feature.size(1)).transpose(1, 2)).squeeze(1)  # batch_size

			# possibility_score_mask = possibility_score == 0
			# if self.training:
			# 	label_mask = (label > 0.01).bool()
			# 	possibility_score = possibility_score.masked_fill(possibility_score_mask & ~label_mask, 1e-9)
			# else:
			# 	possibility_score = possibility_score.masked_fill(possibility_score_mask, 1e-9)

			# possibility_score = possibility_score.masked_fill(possibility_score_mask, -1e4)
			# hr_abstract_prob = hr_abstract_topk_mask.unsqueeze(1) * self.possibility_codebook
			# tail_abstract_prob = (tail_abstract_topk_mask.unsqueeze(1) * self.possibility_codebook.T).transpose(1, 2)
			# 太慢了
			# for i in range(hr_abstract_prob.size(0)):
			# 	for j in range(tail_abstract_prob.size(0)):
			# 		possibility_score[i,j] = torch.mean(hr_abstract_prob[i] * tail_abstract_prob[j])
			# 显存爆炸
			# possibility_score = torch.mean(hr_abstract_prob.unsqueeze(1) * tail_abstract_prob.unsqueeze(0), dim=[-1, -2])
		else:
			# intermediate = torch.bmm(hr_abstract_feature.unsqueeze(1), pc.expand(hr_abstract_feature.size(0), self.p.class_num, self.p.class_num))  # batch_size x 1 x K
			intermediate = torch.bmm(hr_abstract_feature.unsqueeze(1), pc)  # batch_size x 1 x K

        	# 再与 tail_mask 进行乘法： batch_size x 1 x K * batch_size x K x t -> batch_size x 1 x t
			possibility_score = torch.bmm(intermediate, tail_abstract_feature.unsqueeze(0).expand(hr_abstract_feature.size(0), tail_abstract_feature.size(0), tail_abstract_feature.size(1)).transpose(1, 2)).squeeze(1)  # batch_size


		return tucker_logits, possibility_score

class TransE_score(torch.nn.Module):
	def __init__(self, params=None):
		super(TransE_score, self).__init__()
		self.p = params


	def forward(self, sub, rel, x, label,  pos_neg_ent=None):

		sub_emb, rel_emb, all_ent	= sub, rel, x
		
		obj_emb				= sub_emb + rel_emb


		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		original_score = x

		if pos_neg_ent != None:
			
			row_idx = torch.arange(x.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			x = x[row_idx, pos_neg_ent]

		return x, original_score

class DistMult_score(torch.nn.Module):
	def __init__(self, params=None):
		super(DistMult_score, self).__init__()
		self.p = params
		
		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

		self.hr_classificer = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		self.t_classificer = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		# self.class_connect_probability = torch.nn.Parameter(torch.rand((self.p.class_num, self.p.class_num)))
		self.class_connect_probability = get_param((self.p.class_num, self.p.class_num))
		
	def forward(self, sub, rel, x, label, pos_neg_ent=None):

		sub_emb, rel_emb, all_ent	= sub, rel, x
		obj_emb				= sub_emb * rel_emb
		
		
		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)
		original_score = x


		hr_abstract_feature = self.hr_classificer(obj_emb)
		all_ent_abstract_feature = self.t_classificer(all_ent)

		if self.p.binary_classifier_mask:
			_, hb = torch.topk(hr_abstract_feature, self.p.topk, dim=-1)
			_, tb = torch.topk(all_ent_abstract_feature, self.p.topk, dim=-1)

			hb = torch.zeros_like(hr_abstract_feature).scatter(-1, hb, 1)
			tb = torch.zeros_like(all_ent_abstract_feature).scatter(-1, tb, 1)
			score = hb @ self.class_connect_probability @ all_ent_abstract_feature.T
		else:
			score = hr_abstract_feature @ self.class_connect_probability @ all_ent_abstract_feature.T

		# x = torch.softmax(score, dim=-1)
		x = score

		if pos_neg_ent != None:
			
			row_idx = torch.arange(x.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			x = x[row_idx, pos_neg_ent]

		
		return x, original_score

class ConvE_score(torch.nn.Module):
	def __init__(self, edge_type ,params=None):
		super(ConvE_score, self).__init__()
		self.p = params
		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		self.bn3		= torch.nn.BatchNorm1d(self.p.class_num)
		# self.bn2		= torch.nn.BatchNorm1d(self.p.class_num)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
		self.m_conv2		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)


		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		# flat_sz_h		= int(self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc_conv			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		self.fc_classifier = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		# self.fc			= torch.nn.Linear(self.flat_sz, self.p.class_num)

		self.hr_classificer = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		self.t_classificer = torch.nn.Linear(self.p.embed_dim, self.p.class_num)
		# self.class_connect_probability = torch.nn.Parameter(torch.rand((self.p.class_num, self.p.class_num)))

		self.class_connect_probability = get_param((edge_type, self.p.class_num, self.p.class_num))

		self.W1 = get_param((self.p.embed_dim, self.p.embed_dim))
		# self.W2 = get_param((self.p.embed_dim // 2, self.p.embed_dim))

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		# self.score_linear = torch.nn.Linear(2, 1, False)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		# e1_embed	= e1_embed. view(-1, 1, self.p.class_num)
		# rel_embed	= rel_embed.view(-1, 1, self.p.class_num)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		# stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel, rel_index, x, label, pos_neg_ent=None):
		
		sub_emb, rel_emb, all_ent	= sub, rel, x
		stk_inp1	    = self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp1)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc_conv(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)


		original_score = x @ all_ent.T
		original_score += self.bias.expand_as(original_score)

		# x				= self.bn0(stk_inp2)
		# x				= self.m_conv2(x)
		# x				= self.bn1(x)
		# x				= F.relu(x)
		# x				= self.feature_drop(x)
		# x				= x.view(-1, self.flat_sz)
		# x				= self.fc_classifier(x)
		# x				= self.hidden_drop2(x)
		# x				= self.bn2(x)
		# x				= F.relu(x)

		# x 				= self.hr_classificer(x)
		# x 				= self.bn3(x)
		# x				= F.relu(x)
		
		# all_ent 		= self.t_classificer(all_ent)
		# all_ent			= self.bn3(all_ent)
		# all_ent		 	= F.relu(all_ent)

		# if self.p.binary_classifier_mask:
		# 	_, hb_mask = torch.topk(x, self.p.topk, dim=-1)
		# 	_, tb_mask = torch.topk(all_ent, self.p.topk, dim=-1)

		# 	hb_mask = torch.zeros_like(x).scatter(-1, hb_mask, 1)
		# 	tb_mask = torch.zeros_like(all_ent).scatter(-1, tb_mask, 1)

		# 	batch_probality = self.class_connect_probability[rel_index]
		# 	hb = x * hb_mask
		# 	tb = all_ent * tb_mask

		# 	# B x C ;B x C x C -> B x 1 x C
		# 	score = torch.bmm(hb.unsqueeze(1), batch_probality)
		# 	# B x 1 x C; C x A -> B x A
		# 	score = torch.bmm(score, tb.T.unsqueeze(0).expand(score.size(0), -1, -1)).squeeze()

		# 	# score = hb @ self.class_connect_probability @ tb.T
		# else:
		# 	# batch_probality = self.class_connect_probability[rel_index]
		# 	# # B x C x C -> B x 1 x C
		# 	# score = torch.bmm(x.unsqueeze(1), batch_probality)
		# 	# # B x 1 x C; C x A -> B x A
		# 	# score = torch.bmm(score, all_ent.T.unsqueeze(0).expand(score.size(0), -1, -1)).squeeze()

		# 	# score = x @ self.class_connect_probability @ all_ent.T
		# 	score = torch.full_like(original_score, -1e4)
		
	
		if pos_neg_ent != None:
			
			row_idx = torch.arange(original_score.size()[0], device=x.device)
			row_idx = row_idx.unsqueeze(1).repeat(1,pos_neg_ent.size(-1))
			score = original_score[row_idx, pos_neg_ent]
		else:
			score = original_score

		# return score, original_score
		return score, original_score
