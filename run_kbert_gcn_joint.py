# =================================
# File : run_kbert_gcn_joint.py
# Description :
# Author : ChenX
# CREATE TIME : 2024/3/12 21:12
# =================================
import sys
import os

from imblearn.metrics import geometric_mean_score
from tqdm import tqdm
sys.path.append((os.path.dirname(__file__)))
import argparse
import random
from multiprocessing import Pool

import evaluate
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils import clip_grad_norm
from brain import KnowledgeGraph
from uer.model_builder import build_model
from uer.model_saver import save_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import BertAdam
from uer.utils.seed import set_seed
from uer.utils.tokenizer import *


class SimCSELoss(nn.Module):
    def __init__(self):
        super(SimCSELoss, self).__init__()

    def forward(self, predict: torch.Tensor):
        """
        计算batch内互为负样本，相邻的01为正样本，23为正样本，45为正样本，....
        其他的为负样本
        :param predict: batch_size * 2 的句向量
        :return:
        """
        device = predict.device
        y_true = torch.arange(predict.shape[0], device=device)
        y_shifting = (y_true - y_true % 2 * 2) + 1
        y_true = torch.eq(torch.unsqueeze(y_true, dim=0), torch.unsqueeze(y_shifting, dim=1))
        y_true = torch.where(y_true, torch.ones_like(y_true, dtype=torch.float),
                             torch.zeros_like(y_true, dtype=torch.float))

        sim = F.cosine_similarity(x1=predict.unsqueeze(1), x2=predict.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(predict.shape[0], device=device) * 1e12
        sim = sim / 0.05

        loss = F.cross_entropy(input=sim, target=y_true)
        return loss


class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_channels=num_node_features,
                                    out_channels=16,
                                    heads=2)
        self.conv2 = pyg_nn.GATConv(in_channels=2 * 16,
                                    out_channels=num_classes,
                                    heads=1)

    def forward(self, features, edges, weight):
        x, edge_index = features, edges

        x = self.conv1(x, edge_index, weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BertGCN(nn.Module):
    def __init__(self, args, model):
        super(BertGCN, self).__init__()
        self.args = args
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.simloss = SimCSELoss()
        self.use_vm = False if args.no_vm else True
        # 加载图神经网络
        self.graph = GAT(num_node_features=args.hidden_size, num_classes=args.hidden_size)
        # 加载边权重
        matrix = torch.Tensor(np.load(args.edge_path))
        self.edge_matrix = nn.Parameter(matrix, requires_grad=False)
        self.accuracy_metric = evaluate.load(self.args.accuracy)
        self.precision_metric = evaluate.load(self.args.precision)
        self.recall_metric = evaluate.load(self.args.recall)
        self.f1_metric = evaluate.load(self.args.f1)

    def forward(self, src, label, mask, pos=None, vm=None, other_input_ids=None, other_mask_ids=None,
                other_pos_ids=None, other_vms=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        b, l = src.shape

        emb = self.embedding(src, mask, pos)
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        output = output.reshape(b * l, -1)

        # 获取边的input_id中，任意2个token之间都存在一条假设边
        start = torch.arange(l - 1).to(src.device)
        end = start + 1
        forward = torch.unsqueeze(torch.cat([start, torch.flip(end, dims=[0])], dim=0), dim=0)
        backward = torch.unsqueeze(torch.cat([end, torch.flip(start, dims=[0])], dim=0), dim=0)
        coord = torch.unsqueeze(torch.range(0, b - 1) * l, dim=1).to(src.device).int()
        forward = forward + coord
        forward = torch.unsqueeze(forward.reshape(-1), dim=0)
        backward = backward + coord
        backward = torch.unsqueeze(backward.reshape(-1), dim=0)
        adj_matrix = torch.cat([forward, backward], dim=0)

        # 获取邻接矩阵的权重
        flat_src = src.reshape(-1)
        rows = flat_src[forward.reshape(-1)]
        cols = flat_src[backward.reshape(-1)]
        edge_weights = torch.tensor([self.edge_matrix[row, col].item() for row, col in zip(rows, cols)]).to(
            src.device)

        # 过GAT
        output = self.graph(features=output, edges=adj_matrix, weight=edge_weights)
        output = output.reshape(b, l, -1)

        # 处理other的另一个
        o_emb = self.embedding(src, mask, pos)
        if not self.use_vm:
            vm = None
        o_output = self.encoder(o_emb, mask, vm)
        o_output = o_output.reshape(b * l, -1)
        flat_other_input = other_input_ids.reshape(-1)
        o_rows = flat_other_input[forward.reshape(-1)]
        o_cols = flat_other_input[backward.reshape(-1)]
        o_edge_weights = torch.tensor([self.edge_matrix[row, col].item() for row, col in zip(o_rows, o_cols)]).to(
            src.device)
        o_output = self.graph(features=o_output, edges=adj_matrix, weight=o_edge_weights)
        o_output = o_output.reshape(b, l, -1)

        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
            o_output = torch.mean(o_output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
            o_output = torch.max(o_output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
            o_output = o_output[:, -1, :]
        else:
            output = output[:, 0, :]
            o_output = o_output[:, 0, :]

        # 第一个text分类
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))

        # 第二个text分类
        o_output = torch.tanh(self.output_layer_1(o_output))
        o_logits = self.output_layer_2(o_output)
        o_loss = self.criterion(self.softmax(o_logits.view(-1, self.labels_num)), label.view(-1))

        # 语义相似度损失函数
        predicts = torch.cat([output.unsqueeze(dim=1), o_output.unsqueeze(dim=1)], dim=1).reshape(b * 2, -1)
        s_loss = self.simloss(predict=predicts)

        total_loss = loss + o_loss + s_loss
        return total_loss, logits

    def predict(self, src, label, mask, pos=None, vm=None):
        # Embedding.
        b, l = src.shape

        emb = self.embedding(src, mask, pos)
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        output = output.reshape(b * l, -1)

        # 获取边的input_id中，任意2个token之间都存在一条假设边
        start = torch.arange(l - 1).to(src.device)
        end = start + 1
        forward = torch.unsqueeze(torch.cat([start, torch.flip(end, dims=[0])], dim=0), dim=0)
        backward = torch.unsqueeze(torch.cat([end, torch.flip(start, dims=[0])], dim=0), dim=0)
        coord = torch.unsqueeze(torch.range(0, b - 1) * l, dim=1).to(src.device).int()
        forward = forward + coord
        forward = torch.unsqueeze(forward.reshape(-1), dim=0)
        backward = backward + coord
        backward = torch.unsqueeze(backward.reshape(-1), dim=0)
        adj_matrix = torch.cat([forward, backward], dim=0)

        # 获取邻接矩阵的权重
        flat_src = src.reshape(-1)
        rows = flat_src[forward.reshape(-1)]
        cols = flat_src[backward.reshape(-1)]
        edge_weights = torch.tensor([self.edge_matrix[row, col].item() for row, col in zip(rows, cols)]).to(
            src.device)

        # 过GAT
        output = self.graph(features=output, edges=adj_matrix, weight=edge_weights)
        output = output.reshape(b, l, -1)

        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def add_knowledge_worker(params):
    p_id, sentences, columns, kg, vocab, args = params
    dataset = []

    for item in columns:
        label = int(item["label"])
        text = CLS_TOKEN + item["text"]
        tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], max_length=args.max_length)
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0].astype("bool")

        token_ids = [vocab.get(t) for t in tokens]
        mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

        if "augment_query" in item:
            other_text = CLS_TOKEN + item["augment_query"]
            other_tokens, other_pos, other_vm, _ = kg.add_knowledge_with_vm([other_text],
                                                                            max_length=args.max_length)
            other_tokens = other_tokens[0]
            other_pos = other_pos[0]
            other_vm = vm[0].astype("bool")

            other_token_ids = [vocab.get(t) for t in other_tokens]
            other_mask = [1 if t != PAD_TOKEN else 0 for t in other_tokens]

            dataset.append((token_ids, label, mask, pos, vm, other_token_ids, other_mask, other_pos, other_vm))
        else:
            dataset.append((token_ids, label, mask, pos, vm))
    return dataset


def getargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path",
                        default="chinese-roberta-wwm-ext/pytorch_model.bin",
                        type=str,
                        help="huggingface下的哈工大开源的roberta参数，是用BERT训练的")
    parser.add_argument("--output_model_path", default="models/kbert_gcn_joint.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="chinese-roberta-wwm-ext/vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=False, default="datasets/train.jsonl",
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=False, default="datasets/test.jsonl",
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=False, default="datasets/test.jsonl",
                        help="验证和测试一批数据")
    parser.add_argument("--config_path", default="models/google_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--edge_path", default="datasets/adj.npy", type=str,
                        help="GNN的邻接矩阵的权重分数")
    parser.add_argument("--accuracy", default="evaluate-main/metrics/accuracy", type=str,
                        help="huggingface下的评价指标")
    parser.add_argument("--precision", default="evaluate-main/metrics/precision", type=str,
                        help="huggingface下的评价指标")
    parser.add_argument("--recall", default="evaluate-main/metrics/recall", type=str,
                        help="huggingface下的评价指标")
    parser.add_argument("--f1", default="evaluate-main/metrics/f1", type=str,
                        help="huggingface下的评价指标")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    ### 'first' should be chosen when we use the [cls] encoding to represent a whole sentence ###
    ### 'last' should be chosen when we use rnn encoding to represent a whole sentence ###
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="mean",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    # 这里的dropout白设，后面args = load_hyperparam(args)会从google_config.json再次加载dropout
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=300,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=5,
                        help="打log的步骤")
    parser.add_argument("--valid_steps", type=int, default=200,
                        help="预测的步骤")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    parser.add_argument("--max_grad_norm", type=float, default=3.0,
                        help="梯度裁剪")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", default="Military", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=20, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix", default=0)

    args = parser.parse_args()
    return args


def main():
    args = getargs()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set(list(range(0, 27)))
    args.labels_num = len(labels_set)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # Build classification model.
    model = BertGCN(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def trian_batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, other_input_ids, other_mask_ids,
                           other_pos_ids, other_vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]

            other_input_ids_batch = other_input_ids[i * batch_size: (i + 1) * batch_size, :]
            other_mask_ids_batch = other_mask_ids[i * batch_size: (i + 1) * batch_size, :]
            other_pos_ids_batch = other_pos_ids[i * batch_size: (i + 1) * batch_size, :]
            other_vms_batch = other_vms[i * batch_size: (i + 1) * batch_size]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, other_input_ids_batch, other_mask_ids_batch, other_pos_ids_batch, other_vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            other_input_ids_batch = other_input_ids[instances_num // batch_size * batch_size:, :]
            other_mask_ids_batch = other_mask_ids[instances_num // batch_size * batch_size:, :]
            other_pos_ids_batch = other_pos_ids[instances_num // batch_size * batch_size:, :]
            other_vms_batch = other_vms[instances_num // batch_size * batch_size:, :]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, other_input_ids_batch, other_mask_ids_batch, other_pos_ids_batch, other_vms_batch

    def eavl_batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = ["Military"]
    else:
        spo_files = ["Military"]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    # 读取数据 , 测试
    def read_train_dataset(path, workers_num=1):
        sentences = []
        columns = []
        with jsonlines.open(path, 'r') as reader:
            for row in reader:
                sentences.append(row["text"])
                columns.append(row)
        sentence_num = len(sentences)
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append(
                    (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    #  模型效果评估
    def do_evaluate(args, is_test):
        if is_test:
            dataset = read_train_dataset(args.test_path, workers_num=args.workers_num)
        else:
            dataset = read_train_dataset(args.dev_path, workers_num=args.workers_num)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = torch.LongTensor([example[4] for example in dataset])

        batch_size = args.batch_size
        model.eval()
        predict_list, label_list = [], []
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(
                eavl_batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                loss, logits = model.predict(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                                             vms_batch)
                predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
                predict_list += predictions.cpu().detach().numpy().tolist()
                label_list += label_ids_batch.cpu().detach().numpy().tolist()

        precision = precision_score(y_true=label_list, y_pred=predict_list, average="macro")
        recall = recall_score(y_true=label_list, y_pred=predict_list, average="macro")
        f1 = f1_score(y_true=label_list, y_pred=predict_list, average="macro")
        g_mean = geometric_mean_score(label_list, predict_list, average='macro')

        model.train()
        return precision, recall, f1, g_mean

    # Training phase.
    trainset = read_train_dataset(args.train_path, workers_num=args.workers_num)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    vms = torch.LongTensor(np.array([example[4] for example in trainset]))
    # 然后将 NumPy 数组转换为张量
    other_input_ids = torch.LongTensor([example[5] for example in trainset])
    other_mask_ids = torch.LongTensor([example[6] for example in trainset])
    other_pos_ids = torch.LongTensor([example[7] for example in trainset])
    other_vms = torch.LongTensor(np.array([example[8] for example in trainset]))

    global_step, best_f1 = 0, 0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch, other_input_ids_batch,
                other_mask_ids_batch, other_pos_ids_batch, other_vms_batch) in enumerate(
            trian_batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms, other_input_ids,
                               other_mask_ids, other_pos_ids, other_vms)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)
            other_input_ids_batch = other_input_ids_batch.to(device)
            other_mask_ids_batch = other_mask_ids_batch.to(device)
            other_pos_ids_batch = other_pos_ids_batch.to(device)
            other_vms_batch = other_vms_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch,
                            other_input_ids=other_input_ids_batch, other_mask_ids=other_mask_ids_batch,
                            other_pos_ids=other_pos_ids_batch, other_vms=other_vms_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            if (i+1) % args.report_steps == 0:
                logger.info("global step: {}, epoch: {}, loss: {}", global_step, epoch, loss)
            clip_grad_norm(model.parameters(), args.max_grad_norm)
            loss.backward()
            optimizer.step()

            if global_step % args.valid_steps == 0:
                precision, recall, f1, g_mean = do_evaluate(args, False)
                logger.info("evaluation precision: {}, recall: {}, F1: {},G-mean: {}", precision, recall, f1, g_mean)
                if f1 > best_f1:
                    best_f1 = f1
                    save_model(model, args.output_model_path)
            global_step += 1



        # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    do_evaluate(args, True)


if __name__ == "__main__":
    main()
