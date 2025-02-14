from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask

from torch.nn.init import xavier_uniform_
from torch.nn import Parameter


def build_model(args) -> nn.Module:
    return CustomBertModel(args)

def param_init(size):
    param = Parameter(torch.Tensor(*size))
    xavier_uniform_(param.data)
    return param

@dataclass
class ModelOutput:
    logits: torch.tensor
    possibility_logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    pos_inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.possibility_log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)

        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.hr_cns_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size*2, self.args.class_num),
        )
        self.tail_cns_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size*2, self.args.class_num),
        )
        
        # if self.args.model_plus and self.args.candidate_num > 0:
        #     self.core_tesnor = param_init((self.args.class_num, self.args.class_num, self.args.class_num))
        #     self.r_cns_classifier = torch.nn.Linear(self.config.hidden_size, self.args.class_num)

        self.possibility_codebook = param_init((self.args.rel_num * 2, self.args.class_num, self.args.class_num))
        # self.r_cns_classifier = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.args.class_num),
        #     nn.ReLU(),
        #     nn.Linear(self.args.class_num, self.args.class_num),
        #     nn.ReLU(),
        #     nn.Linear(self.args.class_num, self.args.class_num),
        # )
        # torch.nn.Linear(self.config.hidden_size, self.args.class_num)

        # if self.args.model_plus and self.args.topk > 0:
        #     self.possibility_codebook = param_init((self.args.class_num, self.args.class_num))

        # if self.args.use_self_negative:
        #     self.head_cns_classifier = torch.nn.Linear(self.config.hidden_size, self.args.class_num)

        # self.bnn0 = torch.nn.BatchNorm1d(self.args.class_num)
        # self.bnn1 = torch.nn.BatchNorm1d(self.args.class_num)
        # self.bnn2 = torch.nn.BatchNorm1d(self.args.class_num)

        # self.input_dropout = nn.Dropout(0.1)
        # self.hidden_dropout= nn.Dropout(0.2)
        # self.out_dropout = nn.Dropout(0.1)
        # self.query = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        # self.key = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        # self.value = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, rel_index: torch.tensor,batch_dict: dict) -> dict:
        hr_vector, head_vector ,tail_vector = output_dict['hr_vector'], output_dict['head_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = torch.mm(nn.functional.normalize(hr_vector, dim=1), nn.functional.normalize(tail_vector, dim=1).t())
        if self.args.model_plus:
            if self.args.topk > 0:
                # 获取候选尾实体的向量
                possibility_logits = self._compute_candidate_possibility_logits(hr_vector, rel_index ,tail_vector, logits)
            else:
                possibility_logits = torch.full(logits.size(), -1e4).to(logits.device)
        else:
            possibility_logits = torch.full(logits.size(), -1e4).to(logits.device)

        if self.training:
            logits = logits - torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
            possibility_logits = possibility_logits - torch.zeros(possibility_logits.size()).fill_diagonal_(self.add_margin).to(possibility_logits.device) \
                if self.args.model_plus else possibility_logits
        logits = logits * self.log_inv_t.exp()
        possibility_logits = possibility_logits * self.possibility_log_inv_t.exp() if self.args.model_plus else possibility_logits


        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)
            possibility_logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits, pre_batch_possibility_logits = self._compute_pre_batch_logits(hr_vector, rel_index, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)
            possibility_logits = torch.cat([possibility_logits, pre_batch_possibility_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(torch.nn.functional.normalize(hr_vector, dim=1) * torch.nn.functional.normalize(head_vector, dim=1), dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

            if self.args.model_plus:
                possibility_self_neg_logits = torch.diag(self._compute_candidate_possibility_logits(hr_vector, rel_index, head_vector, self_neg_logits))
                possibility_self_neg_logits = possibility_self_neg_logits.masked_fill(~self_negative_mask, -1e4)
                possibility_self_neg_logits = possibility_self_neg_logits * self.possibility_log_inv_t.exp()
            else:   
                possibility_self_neg_logits = torch.full(self_neg_logits.size(), -1e4).to(hr_vector.device)
            possibility_logits = torch.cat([possibility_logits, possibility_self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'possibility_logits': possibility_logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'pos_inv_t': self.possibility_log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}
    

    def _compute_tucker_logits(self, hr_vector: torch.tensor, tail_vector: torch.tensor, logits: torch.tensor, only_head=False) -> torch.tensor:
        hrp = self.hr_cns_classifier(hr_vector)
        rlp = self.r_cns_classifier(hr_vector)
        # tlp: target linear possibility
        tlp = self.tail_cns_classifier(tail_vector)


        # if self.args.topk != 0:
        #     _, hrp_topk = torch.topk(hrp, self.args.topk, dim=1)
        #     _, tlp_topk = torch.topk(tlp, self.args.topk, dim=1)

        #     hrp_onehot = torch.full((hrp.size(0), self.args.class_num), -1e4).to(hrp.device).scatter_(1, hrp_topk, 1)
        #     tlp_onehot = torch.full((tlp.size(0), self.args.class_num), -1e4).to(tlp.device).scatter_(1, tlp_topk, 1)

        #     if self.args.use_possibility_mask:
        #         hrp = hrp * hrp_onehot
        #         tlp = tlp * tlp_onehot
        #         if only_head:
        #             # 实际上是计算了k行和对应的k列加和
        #             possibility_logits= torch.sum(hrp @ pc * tlp, dim=1) / \
        #                   (torch.mean(pc) * self.args.topk + self.args.topk) 
        #         else:
        #             # 实际上是计算了k行和对应的k列加和
        #             possibility_logits = hrp @ pc @ tlp.t() / \
        #                   (torch.mean(pc) * self.args.topk + self.args.topk)
        #     else:
        #         if only_head:
        #             possibility_logits = torch.sum(hrp @ pc * tlp, dim=1) / \
        #                   (torch.mean(pc) * self.args.topk + self.args.topk) 
        #         else:
        #             possibility_logits = hrp_onehot @ pc @ tlp_onehot.t() / \
        #                   (torch.mean(pc) * self.args.topk + self.args.topk) 
        # else:


        hrp = self.bnn0(hrp)
        # hrp = self.input_dropout(hrp)
        # tlp = self.bnn1(tlp)
        # tlp = self.input_dropout(tlp)

        hrp = hrp.view(-1, 1, hrp.size(1))

        W_mat = torch.mm(rlp, self.possibility_codebook.view(rlp.size(1), -1))
        W_mat = W_mat.view(-1, hrp.size(-1), tlp.size(1))
        # W_mat = self.hidden_dropout(W_mat)

        x = torch.bmm(hrp, W_mat)
        x = x.view(-1, tlp.size(1))
        x = self.bnn2(x)
        # x = self.out_dropout(x)

        if not only_head:
            
            possibility_logits = x @ tlp.t()
            possibility_logits = torch.tanh(possibility_logits)

            if self.training:
                _, possibility_candidate_topk_index = torch.topk(logits, self.args.candidate_num, dim=1)
                _, possibility_candidate_bottomk_index = torch.topk(logits, self.args.candidate_num, dim=1, largest=False)
                possibility_candidate_index = torch.cat([possibility_candidate_topk_index, possibility_candidate_bottomk_index], dim=1).to(hr_vector.device)
                possibility_logits_mask = torch.nn.functional.one_hot(torch.arange(possibility_logits.size(0)), num_classes=possibility_logits.size(1)).to(possibility_logits.device).scatter_(1, possibility_candidate_index, 1)
                # possibility_logits = possibility_logits * possibility_logits_mask
                possibility_logits = possibility_logits.masked_fill(~(possibility_logits_mask.bool()), -1e4)
            else:
                _, possibility_candidate_index = torch.topk(logits, self.args.candidate_num, dim=1)
                possibility_logits_mask = torch.nn.functional.one_hot(torch.arange(possibility_logits.size(0)), num_classes=possibility_logits.size(1)).to(possibility_logits.device).scatter_(1, possibility_candidate_index, 1)
                # possibility_logits_mask = torch.zeros_like(logits).to(logits.device).scatter_(1, possibility_candidate_index, 1)
                possibility_logits = possibility_logits.masked_fill(~(possibility_logits_mask.bool()), -1e4)
            # return possibility_logits, possibility_logits_mask
        else:
            head_possibility_logits = torch.sum(x * tlp, dim=1)
            possibility_logits = torch.sigmoid(head_possibility_logits)
        return possibility_logits

        
    
    def _compute_rerank_attention(self, hr_vector: torch.tensor, tail_vector: torch.tensor) -> torch.tensor:
        q = self.query(hr_vector)
        k = self.key(tail_vector)
        v = self.value(tail_vector)

        attention_scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(k.size(-1)).float())
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_scores, v)
        return attention_output
        
    def _compute_candidate_possibility_logits(self, hr_vector: torch.tensor, rel_index: torch.tensor, tail_vector: torch.tensor,
                                              logits: torch.tensor) -> torch.tensor:

        hr_candidate_vector = hr_vector.unsqueeze(1)
        candidate_tail_abstract_feature_vector = self.tail_cns_classifier(tail_vector)
        hr_candidate_abstract_feature_vector = self.hr_cns_classifier(hr_candidate_vector)

        # TODO 此处一直会多选索引为0的实体，需要修改
        # candidate_tail_abstract_feature_vector = candidate_tail_abstract_feature_vector[candidate_index, :] 
        # candidate_tail_abstract_feature_vector = candidate_tail_abstract_feature_vector.index_select(0, candidate_index.view(-1)).view(logits.size(0), -1, candidate_tail_abstract_feature_vector.size(-1))
        # candidate_tail_abstract_feature_vector = candidate_tail_abstract_feature_vector.gather(1, candidate_index)
        # candidate_index = torch.nn.functional.one_hot(torch.arange(candidate_index.size(0)), num_classes=logits.size(1)).to(logits.device).scatter_(1, candidate_index, 1)
        
        possibility_codebook = torch.tanh(self.possibility_codebook[rel_index])

        # possibility_codebook = torch.sigmoid(possibility_codebook)
        candidate_tail_abstract_feature_vector = torch.sigmoid(candidate_tail_abstract_feature_vector)
        hr_candidate_abstract_feature_vector = torch.sigmoid(hr_candidate_abstract_feature_vector)


        if self.args.topk > 0:
            _, candidate_tail_abstract_feature_vector_topk_index = torch.topk(candidate_tail_abstract_feature_vector, self.args.topk, dim=-1)
            _, hr_candidate_abstract_feature_vector_topk_index = torch.topk(hr_candidate_abstract_feature_vector, self.args.topk, dim=-1)

            # TODO 此处填充值应该为0附近，而非-1e4
            candidate_tail_abstract_feature_vector_topk_mask = torch.full(candidate_tail_abstract_feature_vector.size(), 1e-9).to(logits.device).scatter_(1, candidate_tail_abstract_feature_vector_topk_index, 1)
            hr_candidate_abstract_feature_vector_topk_mask =torch.full(hr_candidate_abstract_feature_vector.size(), 1e-9).to(logits.device).scatter_(2, hr_candidate_abstract_feature_vector_topk_index, 1)
        
            candidate_tail_abstract_mask_vector = candidate_tail_abstract_feature_vector_topk_mask * candidate_tail_abstract_feature_vector
            hr_candidate_abstract_mask_vector = hr_candidate_abstract_feature_vector_topk_mask * hr_candidate_abstract_feature_vector
            # possibility_logits = torch.mean(torch.matmul(hr_candidate_abstract_mask_vector * possibility_codebook, candidate_tail_abstract_mask_vector.transpose(1, 2)).transpose(1, 2), dim=-1).to(logits.device)

            intermediate = torch.bmm(hr_candidate_abstract_mask_vector, possibility_codebook)  # batch_size x 1 x K
            possibility_logits = torch.bmm(intermediate, candidate_tail_abstract_mask_vector.unsqueeze(0).expand(hr_vector.size(0), candidate_tail_abstract_feature_vector.size(0), candidate_tail_abstract_feature_vector.size(1)).transpose(1, 2)).squeeze(1)  # batch_size
        else:
            # possibility_logits = hr_candidate_abstract_feature_vector_topk_mask * possibility_codebook
            # possibility_logits = torch.sum(candidate_tail_abstract_feature_vector_topk_mask.unsqueeze(2) * possibility_logits, dim=-1)
            # merge_mask = hr_candidate_abstract_feature_vector_topk_mask * candidate_tail_abstract_feature_vector_topk_mask
            # possibility_logits = torch.sum(merge_mask * possibility_codebook, dim=-1)
            # possibility_logits = (hr_candidate_abstract_feature_vector_topk_mask @ possibility_codebook @ candidate_tail_abstract_feature_vector_topk_mask.t()).squeeze() / self.args.class_num
            # possibility_logits = torch.mean(torch.matmul(hr_candidate_abstract_feature_vector_topk_mask * possibility_codebook, candidate_tail_abstract_feature_vector_topk_mask.unsqueeze(0).transpose(1, 2)).transpose(1, 2), dim=-1).to(logits.device)

            intermediate = torch.bmm(hr_candidate_abstract_feature_vector.unsqueeze(1), possibility_codebook)  # batch_size x 1 x K
            possibility_logits = torch.bmm(intermediate, candidate_tail_abstract_feature_vector.unsqueeze(0).expand(hr_vector.size(0), candidate_tail_abstract_feature_vector.size(0), candidate_tail_abstract_feature_vector.size(1)).transpose(1, 2)).squeeze(1)  # batch_size

        # if self.training:
        #     _, candidate_topk_index = torch.topk(logits, self.args.candidate_num, dim=-1)
        #     _, candidate_bottomk_index = torch.topk(logits, self.args.candidate_num, dim=-1, largest=False)
        #     candidate_index = torch.cat([candidate_topk_index, candidate_bottomk_index], dim=1).to(logits.device)
        #     candidate_index = torch.nn.functional.one_hot(torch.arange(candidate_index.size(0)), num_classes=logits.size(1)).to(logits.device).scatter_(1, candidate_index, 1)
        # else:
        #     _, candidate_index = torch.topk(logits, self.args.candidate_num, dim=1)
        #     candidate_index = torch.zeros_like(logits).to(logits.device).scatter_(1, candidate_index, 1)

        # possibility_logits = possibility_logits.masked_fill(~(candidate_index.bool()), -1e4)

        # if not self.args.possibility:
        #     possibility_logits = torch.sigmoid(possibility_logits)
        return possibility_logits
        

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  rel_index: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = torch.mm(nn.functional.normalize(hr_vector, dim=1), nn.functional.normalize(self.pre_batch_vectors.clone(), dim=1).t())
        if self.args.model_plus:
            pre_batch_possibility_logits = self._compute_candidate_possibility_logits(hr_vector, rel_index, self.pre_batch_vectors.clone(), pre_batch_logits)
        else:
            pre_batch_possibility_logits = torch.full(pre_batch_logits.size(), -1e4).to(hr_vector.device)
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        pre_batch_possibility_logits *= self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)
            pre_batch_possibility_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits, pre_batch_possibility_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    # output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
