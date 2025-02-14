# %%
from helper import *
# from read_data.data_loader import *
from read_data.read_data_compgcn import read_compgcn
from read_data.read_data_rgcn import read_rgcn
import gzip
import random
import math

# sys.path.append('./')
from model.models import *
from model.rgcn_model import *

from wandb_config import wandb


class Runner(object):

	def __init__(self, params):
		
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		
		pprint(vars(self.p))
		
		self.n_gpu = torch.cuda.device_count()
		
		if self.p.gpu != '-1' and torch.cuda.is_available():

			self.device = torch.device('cuda:0')
			
			self.n_gpu=1

			torch.cuda.manual_seed(self.p.seed)
			torch.cuda.manual_seed_all(self.p.seed)
			
			
			torch.backends.cudnn.deterministic = True
			
		else:
			self.device = torch.device('cpu')

		
		self.model        = self.add_model(self.p.model, self.p.score_func)
		# self.optimizer    = self.add_optimizer(self.model.parameters())



	def add_model(self, model, score_func):
		
		if self.p.read_setting == 'no_negative_sampling': self.read_data = read_compgcn(self.p)
		elif self.p.read_setting == 'negative_sampling': 
			if model == 'transe':
				self.read_data = read_rgcn(self.p, triplet_reverse_loss=True)
			else:
				self.read_data = read_rgcn(self.p)
		
		
		else: raise NotImplementedError('please choose one reaing setting: no_negative_sampling or negative_sampling')

		edge_index, edge_type, self.data_iter, self.feature_embeddings, indices_2hop = self.read_data.load_data()

		

		print('################### model:'+ self.p.model + ' #################')
		print('reading setting: ', self.p.read_setting)

		if  self.p.read_setting == 'no_negative_sampling': 
			if self.p.neg_num != 0: raise ValueError('no negative sampling does not use negative sampling, please the predefined parameter ''neg_num'' be 0')
			print('no negative samples: ', self.p.neg_num)

		elif self.p.read_setting == 'negative_sampling': 
			if self.p.neg_num <= 0: raise ValueError('use negative sampling, please the predefined parameter ''neg_num'' to be larger than 0')
			
			if self.p.use_all_neg_samples:
				print('use all possible negative samples ')
			else:
				print('negative samples: ', self.p.neg_num)
		

		model = BaseModel(edge_index, edge_type, params=self.p, feature_embeddings=self.feature_embeddings, indices_2hop=indices_2hop)

		model.to(self.device)
		if self.n_gpu > 1:
			model = torch.nn.DataParallel(model)

		
		return model

	def add_optimizer(self, parameters):
		
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	

	def save_model(self, save_path):
		
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		param  = state['args']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		state_dict_copy = dict()
		for k in state_dict.keys():
			
			state_dict_copy[k] = state_dict[k]
		
		print('load params: ', param)
		self.model.load_state_dict(state_dict_copy)
		

	def evaluate(self, split, epoch, mode, f_test):
		self.logger.info('[Epoch {} {} {} {}]: Evaluating...'.format(epoch, split, mode ,'tail_batch'))
		left_results  = self.predict(split=split, mode='tail_batch')

		self.logger.info('[Epoch {} {} {} {}]: Evaluating...'.format(epoch, split, mode ,'head_batch'))
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		# self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		# self.logger.info('[Epoch {} {}]: MR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mr'], results['right_mr'], results['mr']))
		# self.logger.info('[Epoch {} {}]: Hits@1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_hits@1'], results['right_hits@1'], results['hits@1']))
		# self.logger.info('[Epoch {} {}]: Hits@3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_hits@3'], results['right_hits@3'], results['hits@3']))
		# self.logger.info('[Epoch {} {}]: Hits@10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_hits@10'], results['right_hits@10'], results['hits@10']))
		if mode == 'val':
			self.file_val_result.write(str(results['mrr'])+'\n')
			self.file_val_result.flush()
		if mode == 'test':
			self.file_test_result.write(str(results['mrr'])+'\n')
			self.file_test_result.flush()
			
		if mode == 'test' and self.best_update == 1:
			f_test.write('left right MRR: '+str(results['left_mrr']) + '\t' +str(results['right_mrr']) + '\t' + str(results['mrr']) + '\n' )
			f_test.write('left right MR: '+str(results['left_mr']) + '\t' +str(results['right_mr']) + '\t' + str(results['mr']) + '\n' )
			f_test.write('left right hits@1: '+str(results['left_hits@1']) + '\t' +str(results['right_hits@1']) + '\t' + str(results['hits@1']) + '\n' )
			f_test.write('left right hits@3: '+str(results['left_hits@3']) + '\t' +str(results['right_hits@3']) + '\t' + str(results['hits@3']) + '\n' )
			f_test.write('left right hits@10: '+str(results['left_hits@10']) + '\t' +str(results['right_hits@10']) + '\t' + str(results['hits@10']) + '\n' )
			f_test.write('****************************************************\n')
			f_test.flush()
			self.best_update = 0

		return results

	def predict(self, split='valid', mode='tail_batch'):
		
		self.model.eval()
		
		with torch.no_grad():
			results = {}
			based_results = {}
			tucker_results = {}
			possibility_results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_data.read_batch(batch, split, self.device)

				x, r			= self.model.forward(sub, rel, obj)
			
				pred, _	 = self.model.get_loss(x, r, sub, rel, obj, label, pos_neg_ent=None)

				# b_range			= torch.arange(pred.size()[0], device=self.device)
				# target_pred		= pred[b_range, obj]
				# pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				# pred[b_range, obj] 	= target_pred
				
				# ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
				
				# ranks 			= ranks.float()
				# results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				# results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				# results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				# for k in range(10):
				# 	results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)
				if isinstance(pred, tuple):
					final_pred, based_pred, tucker_pred, prossibility_pred = pred
					self.compute_results(final_pred, obj, label, results)
					self.compute_results(based_pred, obj, label, based_results)
					self.compute_results(tucker_pred, obj, label, tucker_results)
					self.compute_results(prossibility_pred, obj, label, possibility_results)
					
					if step % 500 == 0:
						count = results['count']
						self.logger.info('Step: {}: Eval final based MR:{:.5}, Eval based MR:{:.5},  Eval Tuckermu MR:{:.5}, Eval Prossibility MR:{:.5}\t{}'.format(step, results['mr'] / count, based_results['mr'] / count, tucker_results['mr'] / count, possibility_results['mr'] / count, self.p.name))
						self.logger.info('Step: {}: Eval final based Hits@1:{:.5}, Eval based Hits@1:{:.5},  Eval Tuckermu Hits@1:{:.5}, Eval Prossibility Hits@1:{:.5}\t{}'.format(step, results['hits@1'] / count, based_results['hits@1'] / count, tucker_results['hits@1'] / count, possibility_results['hits@1'] / count, self.p.name))
						self.logger.info('Step: {}: Eval final based Hits@3:{:.5}, Eval based Hits@3:{:.5},  Eval Tuckermu Hits@3:{:.5}, Eval Prossibility Hits@3:{:.5}\t{}'.format(step, results['hits@3'] / count, based_results['hits@3'] / count, tucker_results['hits@3'] / count, possibility_results['hits@3'] / count, self.p.name))
						self.logger.info('Step: {}: Eval final based Hits@10:{:.5}, Eval based Hits@10:{:.5},  Eval Tuckermu Hits@10:{:.5}, Eval Prossibility Hits@10:{:.5}\t{}'.format(step, results['hits@10'] / count, based_results['hits@10'] / count, tucker_results['hits@10'] / count, possibility_results['hits@10'] / count, self.p.name))
				else:	
					self.compute_results(pred, obj, label, results)
					if step % 500 == 0:
						count = results['count']
						self.logger.info('Step: {}: Eval final MR:{:.5}, Eval Tuckermu MR:{:.5}, Eval Prossibility MR:{:.5}\t{}'.format(step, results['mr'] / count, 0.0, 0.0, self.p.name))
						self.logger.info('Step: {}: Eval final Hits@1:{:.5}, Eval Tuckermu Hits@1:{:.5}, Eval Prossibility Hits@1:{:.5}\t{}'.format(step, results['hits@1'] / count, 0.0, 0.0, self.p.name))
						self.logger.info('Step: {}: Eval final Hits@3:{:.5}, Eval Tuckermu Hits@3:{:.5}, Eval Prossibility Hits@3:{:.5}\t{}'.format(step, results['hits@3'] / count, 0.0, 0.0, self.p.name))
						self.logger.info('Step: {}: Eval final Hits@10:{:.5}, Eval Tuckermu Hits@10:{:.5}, Eval Prossibility Hits@10:{:.5}\t{}'.format(step, results['hits@10'] / count, 0.0, 0.0, self.p.name))

		return results

	def compute_results(self, pred, obj, label, results):
		b_range			= torch.arange(pred.size()[0], device=self.device)
		target_pred		= pred[b_range, obj]
		pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
		pred[b_range, obj] 	= target_pred
		
		ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
		
		ranks 			= ranks.float()
		results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
		results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
		results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
		for k in range(10):
			results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

	def run_epoch(self, epoch, val_mrr = 0):
		
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])
		
		count = 0
		final_result, base_result, turck_result, prossibility_result = dict(),dict(), dict(), dict()

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()
			
			# train epochs with warmup
			sub, rel, obj, label, pos_neg_ent = self.read_data.read_batch(batch, 'train', self.device)

			x, r	= self.model.forward(sub, rel, obj)
			pred, loss = self.model.get_loss(x, r, sub, rel, obj, label, pos_neg_ent)

			# inverse_x, inverse_r = self.model.forward(obj, rel, sub)
			# _, inverse_loss = self.model.get_loss(inverse_x, inverse_r, obj, rel, sub, label, pos_neg_ent)
			
			if self.n_gpu > 0:
				# loss = loss.mean() + inverse_loss.mean()
				loss = loss.mean()
			
			if isinstance(pred, tuple):
				final_pred, based_pred, tucker_pred, prossibility_pred = pred
				# basedmu, tuckermu, prossibilitymu = based_pred.mean().item(), tucker_pred.mean().item(), prossibility_pred.mean().item()
				# tuckermus.append(tucker_pred.sum().item()/(self.p.candidate_num * 2 * self.p.batch_size))
				# basedmus.append(based_pred.mean().item())
				# possibilitymus.append(prossibility_pred.mean().item())
				self.compute_results(final_pred, obj, label, final_result)
				self.compute_results(based_pred, obj, label, base_result)
				self.compute_results(tucker_pred, obj, label, turck_result)
				self.compute_results(prossibility_pred, obj, label, prossibility_result)
				if step % 500 == 0:
					count = float(base_result['count'])
					self.logger.info('[Epoch:{}| {}]: Train final based MRR:{:.5}, Train based MRR:{:.5}, Train Tuckermu MRR:{:.5}, Train Prossibility MRR:{:.5}\t{}'.format(epoch, step, final_result['mrr'] / count, base_result['mrr'] / count, turck_result['mrr'] / count, prossibility_result['mrr'] / count, self.p.name))
					self.logger.info('[Epoch:{}| {}]: Train final based Hits@1:{:.5}, Train based Hits@1:{:.5}, Train Tuckermu Hits@1:{:.5}, Train Prossibility Hits@1:{:.5}\t{}'.format(epoch, step, final_result['hits@1'] / count, base_result['hits@1'] / count, turck_result['hits@1'] / count, prossibility_result['hits@1'] / count, self.p.name))
					self.logger.info('[Epoch:{}| {}]: Train final based Hits@3:{:.5}, Train based Hits@3:{:.5}, Train Tuckermu Hits@3:{:.5}, Train Prossibility Hits@3:{:.5}\t{}'.format(epoch, step, final_result['hits@3'] / count, base_result['hits@3'] / count, turck_result['hits@3'] / count, prossibility_result['hits@3'] / count, self.p.name))
					self.logger.info('[Epoch:{}| {}]: Train final based Hits@10:{:.5}, Train based Hits@10:{:.5}, Train Tuckermu Hits@10:{:.5}, Train Prossibility Hits@10:{:.5}\t{}'.format(epoch, step, final_result['hits@10'] / count, base_result['hits@10'] / count, turck_result['hits@10'] / count, prossibility_result['hits@10'] / count, self.p.name))
					
			else:
				self.compute_results(pred, obj, label, final_result)
				# self.compute_results(inverse_pred, sub, sub, final_result)
				if step % 500 == 0:
					count = float(final_result['count'])
					self.logger.info('[Epoch:{}| {}]: Train final MRR:{:.5}, Train Tuckermu MRR:{:.5}, Train Prossibility MRR:{:.5}\t{}'.format(epoch, step, final_result['mrr'] / count , 0.0, 0.0, self.p.name))
					self.logger.info('[Epoch:{}| {}]: Train final Hits@1:{:.5}, Train Tuckermu Hits@1:{:.5}, Train Prossibility Hits@1:{:.5}\t{}'.format(epoch, step, final_result['hits@1'] / count , 0.0, 0.0, self.p.name))	
					self.logger.info('[Epoch:{}| {}]: Train final Hits@3:{:.5}, Train Tuckermu Hits@3:{:.5}, Train Prossibility Hits@3:{:.5}\t{}'.format(epoch, step, final_result['hits@3'] / count , 0.0, 0.0, self.p.name))
					self.logger.info('[Epoch:{}| {}]: Train final Hits@10:{:.5}, Train Tuckermu Hits@10:{:.5}, Train Prossibility Hits@10:{:.5}\t{}'.format(epoch, step, final_result['hits@10'] / count , 0.0, 0.0, self.p.name))

			loss.backward()
			
			self.optimizer.step()
			losses.append(loss.item())

			# if step % 5000 == 0:
			# 	# count   = float(results['count'])
			# 	# ave_train_mrr = round(results ['mrr'] /count, 5)
			# 	ave_train_mrr = 0.0

			# 	if self.model.model.p.ratio is not None:
			# 		self.logger.info('[Epoch:{}| {}]: Train Loss:{:.5}, Train MRR:{:.5}, Val MRR:{:.5}, Train Ratio:{:.5}\t{}'.format(epoch, step, np.mean(losses), ave_train_mrr, self.best_val_mrr, self.model.model.p.ratio, self.p.name))
			# 	else:
			# 		self.logger.info('[Epoch:{}| {}]: Train Loss:{:.5}, Train MRR:{:.5}, Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), ave_train_mrr, self.best_val_mrr, self.p.name))
				# wandb.log({
				# 	'Epoch': epoch,
				# 	'Step': step,
				# 	'Train Loss': np.mean(losses),
				# 	'Train MRR': ave_train_mrr,
				# 	'Val MRR': self.best_val_mrr
				# })

		# if not self.p.loss_alpha_update:
		# 	self.model.tucker_alpha = torch.FloatTensor(basedmus).mean().item() / (torch.FloatTensor(tuckermus).mean().item() + 1e-8)
		# 	self.model.possibility_alpha = torch.FloatTensor(basedmus).mean().item() / (torch.FloatTensor(possibilitymus).mean().item() + 1e-8)

		loss = np.mean(losses)

		count = final_result['count']
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}  TrainTuckerAlpha:{:.4}  TrainPossibilityAlpha:{:.4}\n'.format(epoch, loss, self.model.tucker_alpha.item(), self.model.possibility_alpha.item()))
		self.logger.info('[Epoch:{}]:  Train based MRR:{:.5}, based Hits@1:{:.5}, based Hits@3:{:.5}, based Hits@10:{:.5}\n'.format(epoch, final_result['mrr']/count, final_result['hits@1']/count, final_result['hits@3']/count, final_result['hits@10']/count))
		wandb.log({
			"Epoch": epoch,
			"Loss": loss,
			'Val MRR': self.best_val_mrr
		})
		return loss
	
	def pretrain_tucker_poss(self):
		"""
		1) freeze base emb (model.init_embed), freeze base model aggregator
		2) only train tucker_emb + AbstractFeature_score + the core tensor, codebook
		3) use subgraph/neighbor triple as (h, r, t, label=1/0) => BCE
		"""
		for param in self.model.parameters():
			param.requires_grad = False

		# 只解冻 T+P
		for param in self.model.Abstraction_score.parameters():
			param.requires_grad = True
		self.model.abstract_entity_embedding.weight.requires_grad = True
		self.model.abstract_relation_embedding.weight.requires_grad = True

		# re-init optimizer to only optimize T+P
		pretrain_params = []
		pretrain_params.append(self.model.abstract_entity_embedding.weight)
		pretrain_params.append(self.model.abstract_relation_embedding.weight)
		for name, param in self.model.Abstraction_score.named_parameters():
			if param.requires_grad:
				pretrain_params.append(param)
		# create a new optimizer
		opt = torch.optim.Adam(pretrain_params, lr=self.p.lr)  # or a separate lr

		# Simple BCE
		bceloss = torch.nn.BCEWithLogitsLoss()

		self.model.train()
		for ep in range(self.p.pretrain_epochs):
			total_loss = 0.0
			train_iter = iter(self.data_iter['train'])
			pretrain_result = {}
			tucker_result = {}
			possibility_result = {}
			for step, batch in enumerate(train_iter):
				# batch => (h, r, t, label=0/1) ...
				sub, rel, obj, label, pos_neg_ent = self.read_data.read_batch(batch, 'train', self.device)

				opt.zero_grad()

				# get T+P embedding
				# sub_emb = model.get_tucker_emb(h)
				# rel_emb => if tucker also needs separate relation emb, define similarly
				# but you might just do "relation_vector = model.tucker_emb_for_rel(r)" if separate,
				# or reuse base, or simply pass a dummy?

				sub_emb  = self.model.abstract_entity_embedding(sub)
				rel_emb  = self.model.abstract_relation_embedding(rel)
				tail_emb = self.model.abstract_entity_embedding.weight

				# Then call model.Abstraction_score
				tucker_logits, poss_score = self.model.Abstraction_score(sub_emb, rel_emb, tail_emb, label)
				# consistency_loss = (tucker_logits - poss_score).pow(2).mean()

				# combine
				final_score = tucker_logits + poss_score
				# # _, final_score_mask = torch.topk(final_score, k=self.p.num_ent // (self.p.pretrain_epochs - ep), dim=1)
				# final_score_mask = torch.topk(final_score, k=self.p.num_ent // 10, dim=1)[1]
				# label_index1, label_index2 = torch.where(label > 0.1)


				# final_score_mask = torch.zeros_like(final_score).scatter_(1, final_score_mask, 1)
				# final_score = final_score.masked_fill(~final_score_mask.bool() & ~label_mask, -1e9)

				# # if you want scale alpha => final_score = alpha*(tucker_logits + poss_score)
				# # loss = bceloss(torch.sigmoid(final_score), label) + consistency_loss
				# loss = bceloss(final_score, label)
				row_id = torch.arange(final_score.size(0), device=final_score.device).unsqueeze(1)

				batch_candidate = torch.argsort(torch.argsort(final_score, dim=1, descending=True), dim=1, descending=False)[row_id, obj].max()

				# _, topk_idx = torch.topk(final_score, k=self.p.num_ent // 10, dim=-1)

				_, topk_idx = torch.topk(final_score, k=batch_candidate, dim=-1)

				# union_idx = torch.cat([topk_idx, real_tail_idx.unsqueeze(1)], dim=1)

				# union_idx_sorted, _ = union_idx.sort(dim=1)

				row_id = row_id.expand_as(topk_idx)  
				# shape [batch_size, candidate_num+1]

				gather_final = final_score[row_id, topk_idx]  # [batch_size, candidate_num+1]
				gather_label = label[row_id, topk_idx]        # [batch_size, candidate_num+1]

				# 6) BCEwithLogits => gather_final is raw logits
				loss = bceloss(gather_final, gather_label)
				loss = loss + 1e-4 * torch.norm(self.model.Abstraction_score.possibility_codebook.data, p=2) + \
						1e-4 * torch.norm(self.model.Abstraction_score.core_tensor.data, p=2)

				loss.backward()
				opt.step()

				self.compute_results(final_score, obj, label, pretrain_result)
				self.compute_results(tucker_logits, obj, label, tucker_result)
				self.compute_results(poss_score, obj, label, possibility_result)

				total_loss += loss.item()

			count = float(pretrain_result['count'])
			self.logger.info('[Pretrain Epoch:{}| {}]: MRR:{:.5}, Hits@1:{:.5}, Hits@3:{:.5}, Hits@10:{:.5}\t{}'.format(ep, step, pretrain_result['mrr'] / count, pretrain_result['hits@1'] / count, pretrain_result['hits@3'] / count, pretrain_result['hits@10'] / count, self.p.name))
			self.logger.info('[Pretrain Epoch:{}| {}]: Tucker MRR:{:.5}, Tucker Hits@1:{:.5}, Tucker Hits@3:{:.5}, Tucker Hits@10:{:.5}\t{}'.format(ep, step, tucker_result['mrr'] / count, tucker_result['hits@1'] / count, tucker_result['hits@3'] / count, tucker_result['hits@10'] / count, self.p.name))
			self.logger.info('[Pretrain Epoch:{}| {}]: Possibility MRR:{:.5}, Possibility Hits@1:{:.5}, Possibility Hits@3:{:.5}, Possibility Hits@10:{:.5}\t{}'.format(ep, step, possibility_result['mrr'] / count, possibility_result['hits@1'] / count, possibility_result['hits@3'] / count, possibility_result['hits@10'] / count, self.p.name))
			self.logger.info(f"[Pretrain epoch:{ep}] average loss={total_loss/(step+1):.4f}")
			self.logger.info("--"*50)
		
		for param in self.model.parameters():
			param.requires_grad = True

		for param in self.model.Abstraction_score.parameters():
			param.requires_grad = False
		self.model.abstract_entity_embedding.weight.requires_grad = False
		self.model.abstract_relation_embedding.weight.requires_grad = False
		self.model.tucker_alpha.requires_grad = False
		self.model.possibility_alpha.requires_grad = False
		self.model.p.model_plus = False

		opt = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)  # or a separate lr

		for ep in range(self.p.pretrain_epochs):
			losses = []
			train_iter = iter(self.data_iter['train'])
			final_result = {}

			for step, batch in enumerate(train_iter):
				opt.zero_grad()
				
				sub, rel, obj, label, pos_neg_ent = self.read_data.read_batch(batch, 'train', self.device)
				

				x, r	= self.model.forward()
				pred, loss = self.model.get_loss(x, r, sub, rel, obj, label, pos_neg_ent)
				
				if self.n_gpu > 0:
					loss = loss.mean()
				
				if step % 500 == 0:
					self.compute_results(pred, obj, label, final_result)
					count = float(final_result['count'])
					self.logger.info('[Pretrain Epoch:{}| {}]: Train based MRR:{:.5}\t{}'.format(ep, step, final_result['mrr'] / count, self.p.name))
					self.logger.info('[Pretrain Epoch:{}| {}]: Train based Hits@1:{:.5}\t{}'.format(ep, step, final_result['hits@1'] / count, self.p.name))	
					self.logger.info('[Pretrain Epoch:{}| {}]: Train based Hits@3:{:.5}\t{}'.format(ep, step, final_result['hits@3'] / count, self.p.name))
					self.logger.info('[Pretrain Epoch:{}| {}]: Train based Hits@10:{:.5}\t{}'.format(ep, step, final_result['hits@10'] / count, self.p.name))

				loss.backward()
				
				opt.step()
				losses.append(loss.item())

			self.logger.info(f"[Pretrain epoch:{ep}] average loss={np.mean(losses):.4f}")


	def unfreeze_base_after_pretrain(self):
		"""
		解冻剩余部分(base aggregator, base embedding)
		用于在预训练结束后，再进行全模型训练
		"""
		# 模型全部解冻
		for param in self.model.parameters():
			param.requires_grad = True
		self.model.p.model_plus = True

		# 重新创建 optimizer 包含全部参数
		self.optimizer = self.add_optimizer(self.model.parameters())

		self.logger.info("=== Base aggregator & base embedding are unfreezed now! Ready for full training ===")


	def fit(self):
		
		self.best_val_mrr, self.best_val, self.best_test, self.best_epoch, val_mrr = 0., {}, {},0, 0.
		save_path = os.path.join(self.p.output_dir, self.p.name+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed))

		if not os.path.exists(self.p.output_dir):
			os.mkdir(self.p.output_dir)
		###########debug

		kill_cnt = 0
		f_test = open(os.path.join(self.p.output_dir, self.p.name + '_mrr_best_scores_'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'_'+str(self.p.batch_size)+'_'+str(self.p.class_num)+'.txt'), 'w')
		f_train_result =  open(os.path.join(self.p.output_dir, self.p.name + '_mrr_train_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'_'+str(self.p.batch_size)+'_'+str(self.p.class_num)+'.txt'), 'w')
		f_val_result = open(os.path.join(self.p.output_dir, self.p.name + '_mrr_val_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'_'+str(self.p.batch_size)+'_'+str(self.p.class_num)+'.txt'), 'w')
		f_test_result = open(os.path.join(self.p.output_dir, self.p.name + '_mrr_test_scores'+str(self.p.lr)+'_'+str(self.p.hid_drop)+'_'+str(self.p.l2)+'_'+str(self.p.seed)+'_'+str(self.p.batch_size)+'_'+str(self.p.class_num)+'.txt'), 'w')

		

		self.file_train_result = f_train_result
		self.file_val_result = f_val_result
		self.file_test_result = f_test_result

		# for pretrain_epoch in range(self.p.pretrain_epochs):
		# 	self.pretrain_tucker_poss(self.model, self.data_iter['train'], self.optimizer, self.device, alpha=0.1, epochs=200)

		if self.p.pretrain_epochs > 0:
			self.pretrain_tucker_poss()
			self.logger.info("Pretrain Finished!")
			self.unfreeze_base_after_pretrain()
		else:
			self.optimizer = self.add_optimizer(self.model.parameters())

		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)  

			if epoch % self.p.evaluate_every == 0: 
				self.best_update = 0
				val_results = self.evaluate('valid', epoch, 'val',f_test)

				if val_results['mrr'] > self.best_val_mrr:
					self.best_update = 1
					self.best_val	   = val_results
					self.best_val_mrr  = val_results['mrr']
					self.best_epoch	   = epoch
					self.save_model(save_path)   
					kill_cnt = 0
			

				else:
					kill_cnt += 1
					if kill_cnt % 10 == 0 and self.p.gamma > 5:
						self.p.gamma -= 5 
						self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
					if kill_cnt > self.p.kill_cnt: 
						self.logger.info("Early Stopping!!")
						break
				
				self.logger.info('Evaluating on Test data')
				test_results = self.evaluate('test', epoch, 'test', f_test)

				
				# self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr)) #debug
				self.logger.info('[Epoch {}] Test: MRR: {:.5}, Hit@1: {:.5}, Hit@3: {:.5}, Hit@10: {:.5}, Val MRR: {:.5}\n\n'.format(epoch, test_results['mrr'], test_results['hits@1'],test_results['hits@3'],test_results['hits@10'],self.best_val_mrr)) #debug

				# score_weight = self.model.ConvE_score.score_linear.weight.detach()
				wandb.log(
					{
						'Epoch': epoch,
						'Left_MRR': test_results['left_mrr'],
						'Right_MRR': test_results['right_mrr'],
						'MRR': test_results['mrr'],
						'Hit@1':test_results['hits@1'],
						'Hit@3':test_results['hits@3'],
						'Hit@10':test_results['hits@10'],
						# 'SP': score_weight[0][0] / score_weight[0][1]
					}
				)
		
		
# %%		
if __name__ == '__main__':
	# %%	
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('-name',		default='kbgat_classificer_tucker',					help='Set run name for saving/restoring models')
	######################## compgcn
	parser.add_argument('-data',		dest='dataset',         default='WN18RR',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='kbgat',		help='Model Name')
	# parser.add_argument('-score_func',	dest='score_func',	default='classificer_linear',		help='Score Function for Link prediction')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	# parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
	parser.add_argument('-loss_func',	dest='loss_func',	default='bce',		help='Loss Function for Link prediction')
	

	parser.add_argument('-batch',           dest='batch_size',      default=512,    type=int,       help='Batch size')
	parser.add_argument('-kill_cnt',           dest='kill_cnt',      default=100,    type=int,       help='early stopping')
	parser.add_argument("-evaluate_every", type=int, default=1,  help="perform evaluation every n epochs")
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin in the transe score')
	parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=9999999,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=0,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',   default=False,         help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-compgcn_num_bases',	dest='compgcn_num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.2,  	type=float,	help='Dropout after GCN')

	
	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')

	
	###### rgcn
	parser.add_argument('-neg_num',	  	dest='neg_num', 	default=0,   	type=int, 	help='Number of negative samples')
	parser.add_argument('-rgcn_num_bases',	dest='rgcn_num_bases', 	default=None,   	type=int, 	help='Number of basis relation vectors to use in rgcn')
	parser.add_argument('-rgcn_num_blocks',	dest='rgcn_num_blocks', 	default=100,   	type=int, 	help='Number of block relation vectors to use in rgcn layer1')
	parser.add_argument('-no_edge_reverse',	dest='no_edge_reverse', 	action='store_true',   default=False,   	help='whether to use the reverse relation in the aggregation')
    ### use all possible negative samples
	parser.add_argument('-use_all_neg_samples',	dest='use_all_neg_samples', 	action='store_true',   default=False,   	help='whether to use the ')
	
	####### margin loss
	parser.add_argument('-margin',		type=float,             default=10.0,			help='Margin in the marginRankingLoss')
	
	############ config
	parser.add_argument("-data_dir", default='./data',type=str,required=False, help="The input data dir.")
	parser.add_argument("-output_dir", default='./output_test/WN18RR_CompGCN_ConvePlus_ModifyLoss_NoNegative_Corr',type=str,required=False, help="The input data dir.")
	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	
	###adding noise in aggregation
	parser.add_argument("-noise_rate", type=float, default=0., help="the rate of noise edges adding  in aggregation, but not loss")
	parser.add_argument("-all_noise", type=float, default=0., help="use noises to edges in aggregation, 1: only use noise edges, 0: add noise edges")
	

	#####  noise in loss
	parser.add_argument("-loss_noise_rate", type=float, default=0, help="true triplets +  adding noise in loss")
	parser.add_argument("-all_loss_noise", type=float, default=0., help="use noises to triplets in loss, 1: only use noise triplets, 0: add noise triplets")
	parser.add_argument("-strong_noise", action='store_true',   default=False, help="use the stronger noise or not")



	parser.add_argument("-add_triplet_rate", type=float, default=0., help="noise triplets + adding true triplets in loss: the true triplets rate")
	parser.add_argument("-add_triplet_base_noise", type=float, default=0., help="noise triplets + adding true triplets in loss: the noise rate")

	
	parser.add_argument("-left_loss_tri_rate", type=float, default=0, help="removing triplets in loss, the left ratio of true triplest in the loss")
	parser.add_argument("-less_edges_in_aggre", action='store_true',   default=False, help="use less triplets in the aggregation (with the same less triplets in the loss)")
	
	#####  kbgat
	parser.add_argument("-use_feat_input", action='store_true',   default=False,   help="use the node feature as input")
	
	parser.add_argument("-triplet_no_reverse", 	action='store_true',   default=False,   	help='whether to use another vector to denote the reverse relation in the loss')
	parser.add_argument("-no_partial_2hop", 	action='store_true', default=False)
	parser.add_argument("-alpha",  type=float, default=0.2, help="LeakyRelu alphs for SpGAT layer")
	parser.add_argument("-nheads", type=int,  default=2	, help="Multihead attention SpGAT")
	####
	parser.add_argument("-read_setting", default='no_negative_sampling',type=str,required=False, help="different reading setting: no_negative_sampling or negative_sampling")
	parser.add_argument("-contrastive_learning", action='store_true', default=False, help='use contrastive learning')
	parser.add_argument("-class_num", default=100, type=int, help="The classes number of the combine about entities and relations")
	parser.add_argument("-temp", default=0.05, type=float, help="The tempture of contrastive learning")
	parser.add_argument("-topk", default=3, type=int, help="the class number to generate the binary index")
	parser.add_argument("-model_plus", action='store_true', default=False, help='whether to use the abstractive features in the model')
	parser.add_argument("-loss_alpha_update", action='store_true', default=False, help="whether to update the alpha in the loss function")
	parser.add_argument("-candidate_num", type=int, default=20, help="the number of the candidate samples strategy")
	parser.add_argument("-t", type=float, default=0.05, help="the temperature of the softmax function in the contrastive learning")
	parser.add_argument("-pretrain_epochs", type=int, default=200, help="the number of the pretrain epochs")
	parser.add_argument("-gnn_distillation", action='store_true', default=False, help='whether to use the gnn distillation')
	parser.add_argument("-ratio_type", default=str, help="the ratio_type of the gnn distillation, linear, expontial")
	parser.add_argument("-ratio", default=float, help="the ratio_value of the gnn distillation")
	#### ckbgat
	parser.add_argument("-initialization_boudnary", default='Zero-One', type=str, required=False, help="the initialization of the boundary")


	args = parser.parse_args()

	
	if not args.restore:
		args.name = args.name
	
	
	
	
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# %%
	model = Runner(args)
	# %%
	model.fit()
