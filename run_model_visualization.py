import matplotlib.pyplot as plt

from run import Runner
import seaborn as sns
import numpy as np
import argparse
import random
import torch
import os


class Runner_Visualization(Runner):
    def __init__(self, params):
        super().__init__(params)
    
    @torch.no_grad()
    def plot_energy_curve(self, a_e, max_k=50):
        # a_e: [num_entities, K], 实体签名矩阵
        energy = np.square(a_e)  # 实体能量
        sorted_a = np.sort(energy, axis=1)[:, ::-1]  # 每行降序排列
        cumulative_energy = np.cumsum(sorted_a, axis=1) / np.sum(sorted_a, axis=1, keepdims=True)
        mean_energy = np.mean(cumulative_energy, axis=0)  # 平均累积能量
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k+1), mean_energy[:max_k], marker='o', label='Top-k Energy')
        plt.axhline(0.90, color='r', linestyle='--', label='90% Threshold')
        plt.xlabel('Number of Modes (k)')
        plt.ylabel('Cumulative Energy')
        plt.title('Cumulative Energy vs. Top-k Modes')
        plt.legend()
        plt.grid(True)
        # 由于能量平方，所以x轴缩小为一半
        plt.xticks(range(1, max_k+1), [x/2 for x in range(1, max_k+1)])
        plt.show()
        plt.savefig(os.path.join(self.p.output_dir,'energy_curve_all.pdf'), bbox_inches='tight')

    @torch.no_grad()
    def plot_all_energy_curve(self, a_e, max_k=50):
        energy = np.square(a_e)  # 形状: [N, num_entities, K]

        # 2. 对每个样本中每个实体的能量沿模式维度（最后一维）按降序排序
        sorted_energy = np.sort(energy, axis=2)[:, :, ::-1]

        # 3. 计算累积能量并归一化（每个实体的累积能量最后为 1）
        cumulative_energy = np.cumsum(sorted_energy, axis=2) / np.sum(sorted_energy, axis=2, keepdims=True)
        
        # 4. 先在实体维度上求平均，再在新增的第一维（N）上求平均，得到最终的平均能量曲线，形状为 (K,)
        mean_energy = np.mean(cumulative_energy, axis=(0, 1))
        
        # 5. 绘制能量累积曲线
        plt.figure(figsize=(8, 5))

        # 由于能量平方，所以x轴缩小为一半
        plt.plot([x/2 for x in range(1, max_k+1)], mean_energy[:max_k], marker='o', label='Top-k Energy', color=(118/255, 162/255, 185/255))
        plt.axhline(0.85, color='r', linestyle='--', label='85% Threshold')
        plt.xlabel('Number of Modes (k)')
        plt.ylabel('Cumulative Energy')

        highlight_k = 20
        # 注意：由于索引从0开始，因此横轴为20对应的 mean_energy 下标为 19
        highlight_value = mean_energy[highlight_k * 2 - 1]
        # 用散点标出该点
        plt.scatter(highlight_k, highlight_value, color='red', s=100, zorder=5,
                    label=f'Value at k={highlight_k}')
        # 添加标注文字和箭头
        plt.annotate(f"{highlight_value:.2f}",
             xy=(highlight_k, highlight_value),
             xytext=(highlight_k - 1, highlight_value),
             arrowprops=dict(arrowstyle="->", color='red'),
             fontsize=12,
             horizontalalignment='right')



        # plt.title('Cumulative Energy vs. Top-k Modes')
        plt.legend()
        plt.grid(True)
        
        # plt.xticks(range(1, max_k+1), [x/2 for x in range(1, max_k+1)])
        
        plt.show()
        plt.savefig(os.path.join(self.p.output_dir,'energy_curve_all_new2.pdf'), bbox_inches='tight')

    def plot_mode_heatmap(self, P_r, a_e, top_k=20):
        # P_r: [num_relations, K, K], 关系转移矩阵
        # a_e: [num_entities, K], 实体签名矩阵
        mode_importance = np.zeros((20, top_k))
        for r in range(20):
            for k in range(top_k):
                mode_importance[r, k] = np.mean(a_e[:, k] * P_r[r, k, :].sum(axis=0))
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(mode_importance, cmap='inferno', annot=False, fmt='.2f')
        plt.xlabel('Number of Modes (k)')
        plt.ylabel('Relation Index')
        # plt.title('Mode Importance Across Relations')
        plt.savefig(os.path.join(self.p.output_dir,'mode_heatmap_inferno.pdf'), bbox_inches='tight')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-name',        default='visualization',                    help='Set run name for saving/restoring models')
    ######################## compgcn
    parser.add_argument('-data',        dest='dataset',         default='WN18RR',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('-model',        dest='model',        default='kbgat',        help='Model Name')
    parser.add_argument('-model_path',    dest='model_path',       help='Model Path')
    parser.add_argument('-run_mode',        default='train',        help='Mode to run the model: train or visualize')
    parser.add_argument('-score_func',    dest='score_func',    default='conve',        help='Score Function for Link prediction')
    # parser.add_argument('-opn',             dest='opn',             default='sub',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
    parser.add_argument('-loss_func',    dest='loss_func',    default='bce',        help='Loss Function for Link prediction')
    

    parser.add_argument('-batch',           dest='batch_size',      default=512,    type=int,       help='Batch size')
    parser.add_argument('-kill_cnt',           dest='kill_cnt',      default=300,    type=int,       help='early stopping')
    parser.add_argument("-evaluate_every", type=int, default=1,  help="perform evaluation every n epochs")
    parser.add_argument('-gamma',        type=float,             default=40.0,            help='Margin in the transe score')
    parser.add_argument('-gpu',        type=str,               default='0',            help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-epoch',        dest='max_epochs',     type=int,       default=9999999,      help='Number of epochs')
    parser.add_argument('-l2',        type=float,             default=0.0,            help='L2 Regularization for Optimizer')
    parser.add_argument('-lr',        type=float,             default=0.001,            help='Starting Learning Rate')
    parser.add_argument('-lbl_smooth',      dest='lbl_smooth',    type=float,     default=0.1,    help='Label Smoothing')
    parser.add_argument('-num_workers',    type=int,               default=0,                     help='Number of processes to construct batches')
    parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,         help='Seed for randomization')

    parser.add_argument('-restore',         dest='restore',         action='store_true',   default=False,         help='Restore from the previously saved model')
    parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

    parser.add_argument('-compgcn_num_bases',    dest='compgcn_num_bases',     default=-1,       type=int,     help='Number of basis relation vectors to use')
    parser.add_argument('-init_dim',    dest='init_dim',    default=100,    type=int,    help='Initial dimension size for entities and relations')
    parser.add_argument('-gcn_dim',          dest='gcn_dim',     default=200,       type=int,     help='Number of hidden units in GCN')
    parser.add_argument('-embed_dim',    dest='embed_dim',     default=None,   type=int,     help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_layer',    dest='gcn_layer',     default=1,       type=int,     help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop',    dest='dropout',     default=0.1,      type=float,    help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop',      dest='hid_drop',     default=0.2,      type=float,    help='Dropout after GCN')

    
    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2',      dest='hid_drop2',     default=0.3,      type=float,    help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop',     dest='feat_drop',     default=0.3,      type=float,    help='ConvE: Feature Dropout')
    parser.add_argument('-k_w',          dest='k_w',         default=10,       type=int,     help='ConvE: k_w')
    parser.add_argument('-k_h',          dest='k_h',         default=20,       type=int,     help='ConvE: k_h')
    parser.add_argument('-num_filt',      dest='num_filt',     default=200,       type=int,     help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz',        dest='ker_sz',         default=7,       type=int,     help='ConvE: Kernel size to use')

    
    ###### rgcn
    parser.add_argument('-neg_num',          dest='neg_num',     default=0,       type=int,     help='Number of negative samples')
    parser.add_argument('-rgcn_num_bases',    dest='rgcn_num_bases',     default=None,       type=int,     help='Number of basis relation vectors to use in rgcn')
    parser.add_argument('-rgcn_num_blocks',    dest='rgcn_num_blocks',     default=100,       type=int,     help='Number of block relation vectors to use in rgcn layer1')
    parser.add_argument('-no_edge_reverse',    dest='no_edge_reverse',     action='store_true',   default=False,       help='whether to use the reverse relation in the aggregation')
    ### use all possible negative samples
    parser.add_argument('-use_all_neg_samples',    dest='use_all_neg_samples',     action='store_true',   default=False,       help='whether to use the ')
    
    ####### margin loss
    parser.add_argument('-margin',        type=float,             default=10.0,            help='Margin in the marginRankingLoss')
    
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
    
    parser.add_argument("-triplet_no_reverse",     action='store_true',   default=False,       help='whether to use another vector to denote the reverse relation in the loss')
    parser.add_argument("-no_partial_2hop",     action='store_true', default=False)
    parser.add_argument("-alpha",  type=float, default=0.2, help="LeakyRelu alphs for SpGAT layer")
    parser.add_argument("-nheads", type=int,  default=2    , help="Multihead attention SpGAT")
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
    parser.add_argument("-ratio", default=None, help="the ratio of the gnn distillation")

    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runner = Runner_Visualization(args)

    runner.plot_all_energy_curve(torch.tanh(runner.model.Abstraction_score.possibility_codebook).detach().cpu().numpy())

    test_iter = iter(runner.data_iter['train'])

    sub_list = []
    rel_list = []

    for i, batch in enumerate(test_iter):
        sub, rel, obj, label	= runner.read_data.read_batch(batch, 'test', runner.device)
        x, r			= runner.model.forward(sub, rel, obj)
        sub_list.append(sub)
        rel_list.append(rel)

        # if i == 2:
        #     break
    # sub_tensor = torch.cat(sub_list, dim=0)
    # rel_tensor = torch.cat(rel_list, dim=0)

    # a_e = torch.cat([x[sub_tensor,:], r[rel_tensor,:]], dim=1)
    a_e = runner.model.Abstraction_score.head_sub_graph_feature(x)
    a_e = torch.sigmoid(a_e)
    a_e = a_e.detach().cpu().numpy()

    P_r = torch.tanh(runner.model.Abstraction_score.possibility_codebook).detach().cpu().numpy()

    runner.plot_mode_heatmap(P_r, a_e)