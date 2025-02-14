# Knowledge Graph Completion with GNN Distillation & APIM

This repository implements a unified framework for knowledge graph completion (KGC) combining:
1. **GNN Distillation**: Iterative message filtering to prevent over-smoothing
2. **Abstract Probabilistic Interaction Modeling (APIM)**: Structured probabilistic interaction patterns

Built upon [Are_MPNNs_helpful](https://github.com/Juanhui28/Are_MPNNs_helpful) (GNN backbone) and [SimKGC](https://github.com/intfloat/SimKGC) (KGC framework), with key architectural enhancements.

## Environment Setup

### Requirements
- Python 3.9.19
- PyTorch 2.3.0 (CUDA 12.6 recommended)
- DGL 2.2.1 
- Transformers 4.40.2
- NVIDIA Apex (FP16 training)

More detailed requirements can be found in `requirements.txt`.


```bash
conda create -n kgc python=3.9.19
conda activate kgc
pip install -r requirements.txt
```

## Training & Evaluation
GNN-Distill + APIM training and evaluation can be run using the `run.py` script.

```bash
# FB15k-237
nohup python run.py  -model 'compgcn'  -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'FB15k-237'  -lr 0.001 -nheads 2 -batch 512 -embed_dim 200 -gcn_dim 100 -init_dim 100 -k_w 10 -k_h 20  -l2 0. -num_workers 3  -hid_drop 0.3  -pretrain_epochs 0 -candidate_num 0 -topk 20 -gcn_layer 4 -gnn_distillation -ratio_type 'exponential' -ratio 0.74 -output_dir '***'> FB15K237_CompGCN_Layer4_ExpDecay.log 2>&1 &

# WN18RR
nohup python run.py  -model 'compgcn'  -read_setting 'no_negative_sampling' -neg_num 0 -score_func 'conve' -data 'WN18RR'  -lr 0.001 -nheads 2 -batch 512 -embed_dim 200 -gcn_dim 100 -init_dim 100 -k_w 10 -k_h 20  -l2 0. -num_workers 3  -hid_drop 0.3  -pretrain_epochs 0 -candidate_num 0 -topk 20 -gcn_layer 4 -gnn_distillation -ratio_type 'exponential' -ratio 0.74 -output_dir '***'> WN18RR_CompGCN_Layer4_ExpDecay.log 2>&1 &
```

### Key Arguments
- `model`: GNN model architecture (compgcn, rgcn, kbgat)
- `read_setting`: negative sampling setting (in this paper we use no_negative_sampling) 
- `score_func`: scoring function for computing edge scores (in this paper we use conve)
- `data`: knowledge graph dataset (FB15k-237, WN18RR)
- `lr`: learning rate
- `nheads`: number of attention heads
- `batch`: batch size
- `embed_dim`: embedding dimension
- `gcn_dim`: GCN layer dimension
- `init_dim`: initial feature dimension
- `k_w`: the size of conve kernel weight
- `k_h`: the size of conve kernel height
- `l2`: L2 regularization strength
- `num_workers`: number of workers for data loading
- `hid_drop`: dropout rate for hidden layers
- `pretrain_epochs`: number of pretraining epochs (set to 0 for no pretraining)
- `candidate_num`: must be 0 (the relational part of the model is not used in this paper)
- `topk`: number of APIM candidates (in this paper we use 20)
- `gcn_layer`: number of GCN layers (in this paper we use 4)
- `gnn_distillation`: whether to use GNN Distillation 
- `ratio_type`: ratio_type of the GNN Distillation model (linear, exponential)
- `ratio`: ratio of the GNN Distillation model (in this paper we use exponential decay ratio=0.74, linear decay ratio=0.4)
- `output_dir`: output directory for saving model checkpoints and logs




<!-- ## Acknowledgements
This implementation builds upon:

[Are_MPNNs_helpful](https://github.com/Juanhui28/Are_MPNNs_helpful) for GNN architectures

[SimKGC](https://github.com/intfloat/SimKGC) for KGC framework

Funded by National Natural Science Foundation of China (62272129) and Shandong Provincial R&D Program (2023CXPT065). -->