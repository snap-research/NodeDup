import argparse
from itertools import count
from pathlib import Path
import this
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, degree

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset

# from logger import Logger

from utils import get_dataset, do_edge_split, do_edge_split_with_ratio, do_edge_split_with_ratio_large
from torch.nn import BCELoss, BCEWithLogitsLoss

from models import MLP, GCN, SAGE, LinkPredictor, GAT, APPNP_model, JKNet
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists

from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit

import math

import random
import json

from Evaluator import *

from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)

def train(model, predictor, data, split_edge, optimizer, batch_size, encoder_name, dataset):

    edge_index = data.adj_t

    model.train()
    predictor.train()

    criterion = BCEWithLogitsLoss()
    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        if encoder_name == 'mlp':
            h = model(data.x, data=data)
        else:
            h = model(data.x, data.adj_t, data)

        edge = pos_train_edge[perm].t()

        if dataset != "igb-tiny" and dataset != "igb-small":
            neg_edge = negative_sampling(split_edge["full_train"].t().to(data.x.device), num_nodes=data.x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')
        else:
            neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long,
                             device=h.device)

        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        loss = criterion(out, train_label)
        loss.backward()

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, encoder_name, dataset, metric):
    model.eval()
    predictor.eval()

    if encoder_name == 'mlp':
        h = model(data.x)
    else:
        h = model(data.x, data.adj_t)

    results = 0.0
    sum = 0

    pos_test_edges = split_edge['valid']["concat_pos"]
    neg_test_edges = split_edge['valid']["concat_neg"]
    pos_split = split_edge['valid']["split_pos"]
    neg_split = split_edge['valid']["split_neg"]

    pos_preds_all = inference(pos_test_edges, pos_split, h, predictor, batch_size)
    neg_preds_all = inference(neg_test_edges, neg_split, h, predictor, batch_size)

    for node, pos_valid_preds, neg_valid_preds in zip(split_edge['valid']['new'], pos_preds_all, neg_preds_all):
        if split_edge['valid']['new'][node]["positive"].size(1) == 1:
            pos_valid_preds = torch.reshape(pos_valid_preds, (1,1))[0]

        neg_valid_preds = torch.reshape(neg_valid_preds, (1,-1))

        train_results = evaluator.eval({
            'y_pred_pos': pos_valid_preds,
            'y_pred_neg': neg_valid_preds,
        }, metric, True)

        if metric == "auc":
            results += train_results
            sum += 1
        else:
            results += train_results.mean().item() * split_edge['valid']['new'][node]["positive"].size(1)
            sum += split_edge['valid']['new'][node]["positive"].size(1)
    return results/sum

@torch.no_grad()
def inference(concat_edges, split, h, predictor, batch_size):
    predictor.eval()
    preds = [] 
    for perm in DataLoader(range(concat_edges.size(1)), batch_size):
        edges = concat_edges[:, perm]
        pred = predictor(h[edges[0]], h[edges[1]]).squeeze().cpu()
        preds.append(pred)
    preds = torch.cat(preds, dim=0)
    splitted = torch.split(preds, split, dim=0)
    return splitted

def data_augmentation(data, split_edge, method, coldupline, AUGMENT_NUM, augment_nodes):
    ISO=0
    COLD=1
    WARM=2
    #### Calculate the node degree
    neighbor_pos = split_edge['train']['edge'].t()
    neighbor_index = {}
    row, col = neighbor_pos
    for i in range(row.size(0)):
        if row[i].item() in neighbor_index:
            neighbor_index[row[i].item()].append(col[i].item())
        else:
            neighbor_index[row[i].item()] = [col[i].item()]
        if col[i].item() in neighbor_index:
            neighbor_index[col[i].item()].append(row[i].item())
        else:
            neighbor_index[col[i].item()] = [row[i].item()] 

    isolated_nodes = []
    cold_start_nodes = []
    node_category = []
    for key in range(data.x.size(0)):
        if key in neighbor_index:
            if len(list(set(neighbor_index[key]))) <= coldupline:
                cold_start_nodes.append(key)
                node_category.append(COLD)
            else:
                node_category.append(WARM)
        else:
            isolated_nodes.append(key)
            node_category.append(ISO)
            # cold_start_nodes.append(key)
            
    node_category = torch.tensor(node_category)
    data.ISO_mask = (node_category == ISO)
    data.COLD_mask = (node_category == COLD)
    data.WARM_mask = (node_category == WARM)

    ### copy isolated nodes
    if augment_nodes == "cold":
        isolated_nodes = isolated_nodes + cold_start_nodes
    elif augment_nodes == "all":
        isolated_nodes = [i for i in range(data.x.size(0))]

    isolated_nodes = sorted(isolated_nodes)

    if "self_loop" in method:
        isolated_self_loop = torch.stack([torch.tensor(isolated_nodes), torch.tensor(isolated_nodes)], dim=0).t()
        split_edge['train']['edge'] = torch.cat([split_edge['train']['edge']] + [isolated_self_loop] * AUGMENT_NUM, dim=0)
    elif method == "duplicated":
        first_line = []
        second_line = []
        num = 0
        data_x_size = data.x.size(0)
        for aug_i in range(AUGMENT_NUM):
            for i in isolated_nodes:
                second = num + data_x_size
                first_line.append(i)
                first_line.append(second)
                second_line.append(second)
                second_line.append(i)
                num += 1
            ### duplicate isolated nodes
            data.x = torch.cat((data.x, data.x[isolated_nodes]), dim=0)

        ### generate the augmented edges
        first_added = torch.stack([torch.tensor(first_line), torch.tensor(second_line)], dim=0).t()
        split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], first_added), dim=0)
        split_edge['full_train'] = split_edge['train']['edge']
    return data, split_edge


def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='cora')
    parser.add_argument('--predictor', type=str, default='mlp')  ##mean/sum/mlp
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='hits@20', choices=['mrr', 'hits@10', "hits@20", 'hits@30', 'hits@50', 'auc'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--negative_samples', type=int, default=500)
    parser.add_argument('--ratio', type=int, default=10)
    parser.add_argument('--coldupline', type=int, default=2)
    parser.add_argument('--val_rate', type=int, default=5)
    parser.add_argument('--test_rate', type=int, default=10)

    parser.add_argument('--log_dir', type=str, default="results")
    parser.add_argument('--augment', type=str, default='duplicated',
                        choices=['duplicated', 'self_loop','self_loop_dropout','none'])
    parser.add_argument('--augment_times', type=int, default=1)
    parser.add_argument('--augment_nodes', type=str, default="cold") #cold, all

    args = parser.parse_args()
    print(args)

    if args.datasets == "amazon-computers" or args.datasets == "amazon-photos":
        args.val_rate=10
        args.test_rate=40
    elif args.datasets == "igb-tiny" or args.datasets == "igb-small":
        args.val_rate=5
        args.test_rate=10
    else:
        args.val_rate=10
        args.test_rate=20

    saved_file_name =  Path(args.log_dir) / ("sp_augment_" + args.datasets + "-" + args.encoder + "-" + args.predictor + "-" + str(args.metric) + "-" + str(args.patience) + "-" + str(args.negative_samples) + "-" + str(args.augment) + "-" + str(args.augment_times) + "-" + str(args.augment_nodes) + "-" + str(int(time.time()*10000)) + ".txt")

    file = open(saved_file_name, "a+")
    file.write(str(args) + "\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)

    if args.datasets == "igb-tiny" or args.datasets == "igb-small":
        data = torch.load(args.dataset_dir + "/" + args.datasets + ".pkl")
    else:
        dataset = get_dataset(args.dataset_dir, args.datasets)
        data = dataset[0]

    split_edge_neg_path = Path(args.dataset_dir) / (args.datasets + "-" + str(args.val_rate) + "-" + str(args.test_rate) + "-" + str(args.negative_samples) + "neg.pkl")
    if split_edge_neg_path.exists():
        split_edge = torch.load(split_edge_neg_path)
    else:
        if args.datasets == "igb-tiny" or args.datasets == "igb-small":
            split_edge = do_edge_split_with_ratio_large(data, val_ratio=args.val_rate/100.0, test_ratio=args.test_rate/100.0, negative_samples=args.negative_samples)
        else:
            split_edge = do_edge_split_with_ratio(data, val_ratio=args.val_rate/100.0, test_ratio=args.test_rate/100.0, negative_samples=args.negative_samples)
        torch.save(split_edge, split_edge_neg_path)

    split_edge["full_train"] = split_edge['train']['edge']

    import json
    split_edge_degree = Path(args.dataset_dir) / (args.datasets + "-" + str(args.val_rate) + "-" + str(args.test_rate) + "-" + str(args.negative_samples) + "_edge_dict.json")
    if split_edge_degree.exists():
        file = open(split_edge_degree, "r")
        edge_dict = json.load(file)
        file.close()
    else:
        data.train_pos_edge_index = split_edge['train']['edge'].t()
        data.val_pos_edge_index = split_edge['valid']['edge'].t()
        data.test_pos_edge_index = split_edge['test']['edge'].t()

        #### Calculate the node degree
        neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index), dim=1)
        neighbor_index = {}
        row, col = neighbor_pos
        for i in range(row.size(0)):
            if row[i].item() in neighbor_index:
                neighbor_index[row[i].item()].append(col[i].item())
            else:
                neighbor_index[row[i].item()] = [col[i].item()]
            if col[i].item() in neighbor_index:
                neighbor_index[col[i].item()].append(row[i].item())
            else:
                neighbor_index[col[i].item()] = [row[i].item()] 

        edge_dict = {}
        for key in neighbor_index:
            edge_dict[str(key)] = len(list(set(neighbor_index[key])))

        file = open(split_edge_degree, "w")
        file.write(json.dumps(edge_dict))
        file.close()

    data, split_edge = data_augmentation(data, split_edge, args.augment, args.coldupline, args.augment_times, args.augment_nodes)
    
    # concat valid and test edges:
    pos = []
    neg = []
    split_pos = []
    split_neg = []
    for node in split_edge['valid']['new']:
        pos.append(split_edge['valid']['new'][node]["positive"])
        split_pos.append(split_edge['valid']['new'][node]["positive"].size(1))
        neg.append(split_edge['valid']['new'][node]["negative"])
        split_neg.append(split_edge['valid']['new'][node]["negative"].size(1))
    
    split_edge['valid']["concat_pos"] = torch.cat(pos, dim=1)
    split_edge['valid']["concat_neg"] = torch.cat(neg, dim=1)
    split_edge['valid']["split_pos"] = split_pos
    split_edge['valid']["split_neg"] = split_neg

    pos = []
    neg = []
    split_pos = []
    split_neg = []
    for node in split_edge['test']['new']:
        pos.append(split_edge['test']['new'][node]["positive"])
        split_pos.append(split_edge['test']['new'][node]["positive"].size(1))
        neg.append(split_edge['test']['new'][node]["negative"])
        split_neg.append(split_edge['test']['new'][node]["negative"].size(1))
    
    split_edge['test']["concat_pos"] = torch.cat(pos, dim=1)
    split_edge['test']["concat_neg"] = torch.cat(neg, dim=1)
    split_edge['test']["split_pos"] = split_pos
    split_edge['test']["split_neg"] = split_neg

    edge_index = split_edge['train']['edge'].t()
    data.adj_t = edge_index
    input_size = data.x.size()[1]

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        from torch_geometric.utils import undirected
        val_edge_index = undirected.to_undirected(split_edge['valid']['edge'].t())
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        if args.datasets != "collab" and args.datasets != "ppa":
            data.full_adj_t = full_edge_index
        elif args.datasets == "collab" or args.datasets == "ppa":
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data.to(device)

    if args.encoder == 'sage':
        model = SAGE(args.datasets, input_size, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout, cold_dropout=("dropout" in args.augment)).to(device)
    elif args.encoder == 'gcn':
        model = GCN(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, cold_dropout=("dropout" in args.augment)).to(device)
    elif args.encoder == 'appnp':
        model = APPNP_model(input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.encoder == 'gat':
        model = GAT(input_size, args.hidden_channels,
                    args.hidden_channels, 1,
                    args.dropout).to(device)
    elif args.encoder == 'mlp':
        model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout,
                    cold_dropout=("dropout" in args.augment)).to(device)
    elif args.encoder == "jknet":
        model = JKNet(args.datasets, input_size, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator()

    all_saved_results = []
    best_run = 0.0
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        for epoch in range(1, 1 + args.epochs):

            loss = train(model, predictor, data, split_edge,
                         optimizer, args.batch_size, args.encoder, args.datasets)

            results = test(model, predictor, data, split_edge,
                            evaluator, args.batch_size, args.encoder, args.datasets, args.metric)
            
            print(results)
            if results > best_val:
                best_val = results
                cnt_wait = 0
                pretrained_model = {'gnn': model.state_dict(), 'predictor': predictor.state_dict()}
                if results > best_run:
                    best_run = results
            else:
                cnt_wait += 1
            if cnt_wait >= args.patience:
                break

        model.load_state_dict(pretrained_model['gnn'], strict=True)
        predictor.load_state_dict(pretrained_model['predictor'], strict=True)

        model.eval()
        predictor.eval()

        with torch.no_grad():
            if args.encoder == 'mlp':
                h = model(data.x)
            else:
                h = model(data.x, data.full_adj_t)

        # hits_results = 0.0
        # hits_len = 0

        results = {}

        pos_test_edges = split_edge['test']["concat_pos"]
        neg_test_edges = split_edge['test']["concat_neg"]
        pos_split = split_edge['test']["split_pos"]
        neg_split = split_edge['test']["split_neg"]

        pos_preds_all = inference(pos_test_edges, pos_split, h, predictor, args.batch_size)
        neg_preds_all = inference(neg_test_edges, neg_split, h, predictor, args.batch_size)
        #### Calculate the Hits results for each testing node
        for node, pos_test_preds, neg_test_preds in zip(split_edge['test']['new'], pos_preds_all, neg_preds_all):

            if split_edge['test']['new'][node]["positive"].size(1) == 1:
                pos_test_preds = torch.reshape(pos_test_preds, (1,1))[0]

            neg_test_preds = torch.reshape(neg_test_preds, (1,-1))

            test_results = evaluator.eval({
                'y_pred_pos': pos_test_preds,
                'y_pred_neg': neg_test_preds,
            }, args.metric)

            test_mrr = test_results['mrr_list'].mean().item()*100.0
            test_auc = test_results['auc']*100.0

            if str(node) in edge_dict:
                if edge_dict[str(node)] not in results:
                    results[edge_dict[str(node)]] = {}
                    for this_K in [10,20,30,50]:
                        results[edge_dict[str(node)]][f'hits@{this_K}'] = [test_results[f'hits@{this_K}_list'].mean().item()*100.0]
                    # results[edge_dict[str(node)]]["hits"] = [test_hits]
                    results[edge_dict[str(node)]]["mrr"] = [test_mrr]
                    results[edge_dict[str(node)]]["auc"] = [test_auc]
                    results[edge_dict[str(node)]]["number"] = 1
                    results[edge_dict[str(node)]]["hit_num"] = [split_edge['test']['new'][node]["positive"].size(1)]
                else:
                    # results[edge_dict[str(node)]]["hits"].append(test_hits)
                    for this_K in [10,20,30,50]:
                        results[edge_dict[str(node)]][f'hits@{this_K}'].append(test_results[f'hits@{this_K}_list'].mean().item()*100.0)
                    results[edge_dict[str(node)]]["mrr"].append(test_mrr)
                    results[edge_dict[str(node)]]["auc"].append(test_auc)
                    results[edge_dict[str(node)]]["number"] += 1
                    results[edge_dict[str(node)]]["hit_num"].append(split_edge['test']['new'][node]["positive"].size(1))
            else:
                if 0 not in results:
                    results[0] = {}
                    # results[0]["hits"] = [test_hits]
                    for this_K in [10,20,30,50]:
                        results[0][f'hits@{this_K}'] = [test_results[f'hits@{this_K}_list'].mean().item()*100.0]
                    results[0]["mrr"] = [test_mrr]
                    results[0]["auc"] = [test_auc]
                    results[0]["number"] = 1
                    results[0]["hit_num"] = [split_edge['test']['new'][node]["positive"].size(1)]
                else:
                    # results[0]["hits"].append(test_hits)
                    for this_K in [10,20,30,50]:
                        results[0][f'hits@{this_K}'].append(test_results[f'hits@{this_K}_list'].mean().item()*100.0)
                    results[0]["mrr"].append(test_mrr)
                    results[0]["auc"].append(test_auc)
                    results[0]["number"] += 1
                    results[0]["hit_num"].append(split_edge['test']['new'][node]["positive"].size(1))

            # hits_results += train_hits * split_edge['test']['new'][node]["positive"].size(1)
            # hits_len += split_edge['test']['new'][node]["positive"].size(1)

        all_saved_results.append(results)
        ##### Calculate the log bin results
        from results_analysis import result_log
        log_results = result_log(results)

        file = open(saved_file_name, "a")
        for key in sorted(log_results.keys()):
            print_out_str = str(key)
            for this_K in [10,20,30,50]:
                print_out_str += f', hits@{this_K}: ' + str(log_results[key][f'hits@{this_K}']/ log_results[key]["edge_num"])
            print_out_str += ", mrr: " + str(log_results[key]["mrr"]/ log_results[key]["edge_num"])
            print_out_str += ", auc: " + str(log_results[key]["auc"]/ log_results[key]["node_num"])
            print(print_out_str)
            file.write(print_out_str+"\n")
        file.close()

    from results_analysis import sp_results_coldwarm, sp_results_log, save_results
    file = open(saved_file_name, "a")
    print("FINAL RESULTS", file=file)
    group_results, overall_results = sp_results_coldwarm(all_saved_results, args.coldupline)
    save_results(file, group_results, overall_results)
    group_results, overall_results = sp_results_log(all_saved_results)
    save_results(file, group_results, overall_results)
    file.close()

    # file = open(saved_file_name +".json", "w")
    # file.write(json.dumps(all_saved_results))
    # file.close()

if __name__ == "__main__":
    main()