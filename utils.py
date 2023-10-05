import random

import numpy as np
import torch
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges)

from torch_geometric.transforms import NormalizeFeatures, Compose, BaseTransform, ToDevice

import math

from torch_geometric import datasets
import random


def get_dataset(root, name: str):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root, transform=transform)
        return dataset

    pyg_dataset_dict = {
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'pubmed': (datasets.Planetoid, 'Pubmed'),
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo')
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name)

    return dataset

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# From the OGB implementation of SEAL
def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))

        # print(data.train_pos_edge_index.size())
        # print(data.val_pos_edge_index.size())
        # print(data.test_pos_edge_index.size())
        # raise TypeError
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()

    return split_edge

def sample_graph(data, split_edge):
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
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    test_dict = {}
    test_row, test_col = data.test_pos_edge_index
    for i in range(test_row.size(0)):
        if test_row[i].item() in test_dict:
            test_dict[test_row[i].item()]["positive"].append(test_col[i].item())
        else:
            test_dict[test_row[i].item()] = {}
            test_dict[test_row[i].item()]["positive"] = [test_col[i].item()]
        if test_col[i].item() in test_dict:
            test_dict[test_col[i].item()]["positive"].append(test_row[i].item())
        else:
            test_dict[test_col[i].item()] = {}
            test_dict[test_col[i].item()]["positive"] = [test_row[i].item()]

    neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index), dim=1)
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
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    for key in test_dict:
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        test_dict[key]["negative"] = random.sample(pool, k=100)

    test_edges = {}
    for key in test_dict:
        pos_line = [key for i in range(len(test_dict[key]["positive"]))]
        neg_line = [key for i in range(len(test_dict[key]["negative"]))]
        test_edges[key] = {}
        test_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(test_dict[key]["positive"])], dim=0)
        test_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(test_dict[key]["negative"])], dim=0)
    split_edge['test']['new'] = test_edges

    return split_edge

def do_edge_split_with_ratio(data, fast_split=False, val_ratio=0.05, test_ratio=0.1, negative_samples=100):
    # data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    #### randomly split the edges
    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    # print(data.train_pos_edge_index.size())
    # print(data.val_pos_edge_index.size())
    # print(data.test_pos_edge_index.size())
    # raise TypeError

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()


    #### samples the positive pool for 
    # neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index), dim=1)
    # neighbor_index = {}
    # row, col = neighbor_pos
    # for i in range(row.size(0)):
    #     if row[i].item() in neighbor_index:
    #         neighbor_index[row[i].item()].append(col[i].item())
    #     else:
    #         neighbor_index[row[i].item()] = [col[i].item()]
    #     if col[i].item() in neighbor_index:
    #         neighbor_index[col[i].item()].append(row[i].item())
    #     else:
    #         neighbor_index[col[i].item()] = [row[i].item()] 
    
    # for key in neighbor_index:
    #     neighbor_index[key] = list(set(neighbor_index[key]))

    ##### sample positive nodes for validation nodes
    valid_dict = {}
    val_row, val_col = data.val_pos_edge_index
    for i in range(val_row.size(0)):
        if val_row[i].item() in valid_dict:
            valid_dict[val_row[i].item()]["positive"].append(val_col[i].item())
        else:
            valid_dict[val_row[i].item()] = {}
            valid_dict[val_row[i].item()]["positive"] = [val_col[i].item()]
        if val_col[i].item() in valid_dict:
            valid_dict[val_col[i].item()]["positive"].append(val_row[i].item())
        else:
            valid_dict[val_col[i].item()] = {}
            valid_dict[val_col[i].item()]["positive"] = [val_row[i].item()]

    ##### sample positive nodes for testing nodes
    test_dict = {}
    test_row, test_col = data.test_pos_edge_index
    for i in range(test_row.size(0)):
        if test_row[i].item() in test_dict:
            test_dict[test_row[i].item()]["positive"].append(test_col[i].item())
        else:
            test_dict[test_row[i].item()] = {}
            test_dict[test_row[i].item()]["positive"] = [test_col[i].item()]
        if test_col[i].item() in test_dict:
            test_dict[test_col[i].item()]["positive"].append(test_row[i].item())
        else:
            test_dict[test_col[i].item()] = {}
            test_dict[test_col[i].item()]["positive"] = [test_row[i].item()]

    neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index), dim=1)
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
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    ### sample 100 negative nodes for each validation node
    for key in valid_dict:
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        valid_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    valid_edges = {}
    for key in valid_dict:
        pos_line = [key for i in range(len(valid_dict[key]["positive"]))]
        neg_line = [key for i in range(len(valid_dict[key]["negative"]))]
        valid_edges[key] = {}
        valid_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(valid_dict[key]["positive"])], dim=0)
        valid_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(valid_dict[key]["negative"])], dim=0)
    split_edge['valid']['new'] = valid_edges

    ### sample 100 negative nodes for each testing node
    for key in test_dict:
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        test_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    test_edges = {}
    for key in test_dict:
        pos_line = [key for i in range(len(test_dict[key]["positive"]))]
        neg_line = [key for i in range(len(test_dict[key]["negative"]))]
        test_edges[key] = {}
        test_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(test_dict[key]["positive"])], dim=0)
        test_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(test_dict[key]["negative"])], dim=0)
    split_edge['test']['new'] = test_edges

    return split_edge

def maually_split(data, val_ratio, test_ratio):
    from torch_geometric.utils.undirected import to_undirected

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    data.edge_index = data.edge_attr = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    return data

def do_edge_split_with_ratio_large(data, fast_split=False, val_ratio=0.05, test_ratio=0.1, negative_samples=100):
    # data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    #### randomly split the edges
    data = maually_split(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    # print(data.train_pos_edge_index.size())
    # print(data.val_pos_edge_index.size())
    # print(data.test_pos_edge_index.size())
    # raise TypeError

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    # split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    # split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()


    #### samples the positive pool for 
    # neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index), dim=1)
    # neighbor_index = {}
    # row, col = neighbor_pos
    # for i in range(row.size(0)):
    #     if row[i].item() in neighbor_index:
    #         neighbor_index[row[i].item()].append(col[i].item())
    #     else:
    #         neighbor_index[row[i].item()] = [col[i].item()]
    #     if col[i].item() in neighbor_index:
    #         neighbor_index[col[i].item()].append(row[i].item())
    #     else:
    #         neighbor_index[col[i].item()] = [row[i].item()] 
    
    # for key in neighbor_index:
    #     neighbor_index[key] = list(set(neighbor_index[key]))

    ##### sample positive nodes for validation nodes
    valid_dict = {}
    val_row, val_col = data.val_pos_edge_index
    for i in range(val_row.size(0)):
        print(i)
        if val_row[i].item() in valid_dict:
            valid_dict[val_row[i].item()]["positive"].append(val_col[i].item())
        else:
            valid_dict[val_row[i].item()] = {}
            valid_dict[val_row[i].item()]["positive"] = [val_col[i].item()]
        if val_col[i].item() in valid_dict:
            valid_dict[val_col[i].item()]["positive"].append(val_row[i].item())
        else:
            valid_dict[val_col[i].item()] = {}
            valid_dict[val_col[i].item()]["positive"] = [val_row[i].item()]

    ##### sample positive nodes for testing nodes
    test_dict = {}
    test_row, test_col = data.test_pos_edge_index
    for i in range(test_row.size(0)):
        print(i)
        if test_row[i].item() in test_dict:
            test_dict[test_row[i].item()]["positive"].append(test_col[i].item())
        else:
            test_dict[test_row[i].item()] = {}
            test_dict[test_row[i].item()]["positive"] = [test_col[i].item()]
        if test_col[i].item() in test_dict:
            test_dict[test_col[i].item()]["positive"].append(test_row[i].item())
        else:
            test_dict[test_col[i].item()] = {}
            test_dict[test_col[i].item()]["positive"] = [test_row[i].item()]

    neighbor_pos = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index), dim=1)
    neighbor_index = {}
    row, col = neighbor_pos
    for i in range(row.size(0)):
        print(i)
        if row[i].item() in neighbor_index:
            neighbor_index[row[i].item()].append(col[i].item())
        else:
            neighbor_index[row[i].item()] = [col[i].item()]
        if col[i].item() in neighbor_index:
            neighbor_index[col[i].item()].append(row[i].item())
        else:
            neighbor_index[col[i].item()] = [row[i].item()] 
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    ### sample 100 negative nodes for each validation node
    for key in valid_dict:
        print(key)
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        valid_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    valid_edges = {}
    for key in valid_dict:
        print(key)
        pos_line = [key for i in range(len(valid_dict[key]["positive"]))]
        neg_line = [key for i in range(len(valid_dict[key]["negative"]))]
        valid_edges[key] = {}
        valid_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(valid_dict[key]["positive"])], dim=0)
        valid_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(valid_dict[key]["negative"])], dim=0)
    split_edge['valid']['new'] = valid_edges

    ### sample 100 negative nodes for each testing node
    for key in test_dict:
        print(key)
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        test_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    test_edges = {}
    for key in test_dict:
        print(key)
        pos_line = [key for i in range(len(test_dict[key]["positive"]))]
        neg_line = [key for i in range(len(test_dict[key]["negative"]))]
        test_edges[key] = {}
        test_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(test_dict[key]["positive"])], dim=0)
        test_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(test_dict[key]["negative"])], dim=0)
    split_edge['test']['new'] = test_edges

    return split_edge

def create_mask(base_mask, rows, cols):
    return base_mask[rows] & base_mask[cols]

def split_edge_inductive(data, negative_sample_num):
    random.seed(234)
    torch.manual_seed(234)
    from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
    from torch_geometric.utils import subgraph
    from torch_geometric.data import Data
    node_splitter = RandomNodeSplit(num_val=0.0, num_test=0.1)
    new_data = node_splitter(data)

    rows, cols = new_data.edge_index
    old_old_edges = create_mask(new_data.train_mask, rows, cols)
    old_old_ei = new_data.edge_index[:, old_old_edges]

    inference_data = Data(new_data.x, old_old_ei)

    test_nodes_edges = (new_data.train_mask[rows] & new_data.test_mask[cols]) | (
        new_data.test_mask[rows] & new_data.train_mask[cols]) | (new_data.test_mask[rows]&new_data.test_mask[cols])
    test_edges = new_data.edge_index[:, test_nodes_edges]

    test_dict = {}
    test_row, test_col = test_edges
    for i in range(test_row.size(0)):
        if test_row[i].item() in test_dict:
            test_dict[test_row[i].item()]["positive"].append(test_col[i].item())
        else:
            test_dict[test_row[i].item()] = {}
            test_dict[test_row[i].item()]["positive"] = [test_col[i].item()]
        if test_col[i].item() in test_dict:
            test_dict[test_col[i].item()]["positive"].append(test_row[i].item())
        else:
            test_dict[test_col[i].item()] = {}
            test_dict[test_col[i].item()]["positive"] = [test_row[i].item()]
    
    row, col = new_data.edge_index
    neighbor_index = {}
    for i in range(row.size(0)):
        if row[i].item() in neighbor_index:
            neighbor_index[row[i].item()].append(col[i].item())
        else:
            neighbor_index[row[i].item()] = [col[i].item()]
        if col[i].item() in neighbor_index:
            neighbor_index[col[i].item()].append(row[i].item())
        else:
            neighbor_index[col[i].item()] = [row[i].item()] 
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    for key in test_dict:
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        test_dict[key]["negative"] = random.sample(pool, k=negative_sample_num)

    # row, col = test_edges
    # neighbor_index = {}
    # for i in range(row.size(0)):
    #     if new_data.test_mask[row[i].item()]:
    #         if row[i].item() in neighbor_index:
    #             neighbor_index[row[i].item()].append(col[i].item())
    #         else:
    #             neighbor_index[row[i].item()] = [col[i].item()]

    # test_dict = {}
    # for key in neighbor_index.keys():
    #     test_dict[key] = {}
    #     test_dict[key]['positive'] = neighbor_index[key]
    #     pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
    #     test_dict[key]["negative"] = random.sample(pool, k=negative_sample_num)

    test_edges = {}
    for key in test_dict:
        pos_line = [key for i in range(len(test_dict[key]["positive"]))]
        neg_line = [key for i in range(len(test_dict[key]["negative"]))]
        test_edges[key] = {}
        test_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(test_dict[key]["positive"])], dim=0)
        test_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(test_dict[key]["negative"])], dim=0)

    training_only_ei = subgraph(new_data.train_mask, old_old_ei, relabel_nodes=True)[0]
    training_only_x = new_data.x[new_data.train_mask]
    
    from torch_geometric.data import Data, Dataset
    given_data = Data(training_only_x, training_only_ei)
    val_splitter = RandomLinkSplit(0.0, 0.1, is_undirected=True)
    training_data, _, val_data = val_splitter(given_data)
    valid_dict = {}
    val_row, val_col = val_data.edge_label_index[:,val_data.edge_label.bool()]
    for i in range(val_row.size(0)):
        if val_row[i].item() in valid_dict:
            valid_dict[val_row[i].item()]["positive"].append(val_col[i].item())
        else:
            valid_dict[val_row[i].item()] = {}
            valid_dict[val_row[i].item()]["positive"] = [val_col[i].item()]
        if val_col[i].item() in valid_dict:
            valid_dict[val_col[i].item()]["positive"].append(val_row[i].item())
        else:
            valid_dict[val_col[i].item()] = {}
            valid_dict[val_col[i].item()]["positive"] = [val_row[i].item()]

    neighbor_index = {}
    row, col = val_data.edge_index
    for i in range(row.size(0)):
        if row[i].item() in neighbor_index:
            neighbor_index[row[i].item()].append(col[i].item())
        else:
            neighbor_index[row[i].item()] = [col[i].item()]
        if col[i].item() in neighbor_index:
            neighbor_index[col[i].item()].append(row[i].item())
        else:
            neighbor_index[col[i].item()] = [row[i].item()] 
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    ### sample negative nodes for each validation node
    for key in valid_dict:
        if key in neighbor_index:
            pool = [i for i in range(val_data.x.size(0)) if i not in neighbor_index[key] and i != key and i not in valid_dict[key]['positive']]
        else:
            pool = [i for i in range(val_data.x.size(0)) if i != key and i not in valid_dict[key]['positive']]
        valid_dict[key]["negative"] = random.sample(pool, k=negative_sample_num)

    valid_edges = {}
    for key in valid_dict:
        pos_line = [key for i in range(len(valid_dict[key]["positive"]))]
        neg_line = [key for i in range(len(valid_dict[key]["negative"]))]
        valid_edges[key] = {}
        valid_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(valid_dict[key]["positive"])], dim=0)
        valid_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(valid_dict[key]["negative"])], dim=0)

    return training_data, val_data, valid_edges, inference_data, test_edges

def split_edges(edge_index, val_ratio, test_ratio):
    mask = edge_index[0] <= edge_index[1]
    perm = mask.nonzero(as_tuple=False).view(-1)
    perm = perm[torch.randperm(perm.size(0), device=perm.device)]
    num_val = int(val_ratio * perm.numel())
    num_test = int(test_ratio * perm.numel())
    num_train = perm.numel() - num_val - num_test
    train_edges = perm[:num_train]
    val_edges = perm[num_train:num_train + num_val]
    test_edges = perm[num_train + num_val:]
    train_edge_index = edge_index[:,train_edges]
    train_edge_index = torch.cat([train_edge_index, train_edge_index.flip([0])], dim=-1)
    val_edge_index = edge_index[:,val_edges]
    val_edge_index = torch.cat([val_edge_index, val_edge_index.flip([0])], dim=-1)
    test_edge_index = edge_index[:,test_edges]

    return train_edge_index, val_edge_index, test_edge_index

def do_edge_split_with_ratio_large_induc(data, data_name, test_ratio, val_node_ratio, val_ratio, old_old_extra_ratio, negative_samples, split_seed=234):
    # Seed our RNG
    random.seed(split_seed)
    torch.manual_seed(split_seed) 

    from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
    from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges, to_networkx, subgraph)
    from torch_geometric.data import Data, Dataset

    # Assume we only have 1 graph in our dataset
    # assert(len(dataset) == 1)
    # data = dataset[0]

    # Some assertions to help with type inference
    # assert(isinstance(data, Data))
    # assert(data.num_nodes is not None)

    split_edge = {}
    split_edge['train'] = {}
    split_edge['valid'] = {}
    split_edge['test'] = {}

    # sample some negatives to use globally
    # num_negatives = round(test_ratio * data.edge_index.size(1)/2)
    # negative_samples = negative_sampling(data.edge_index, data.num_nodes, num_negatives, force_undirected=True)

    # Step 1: pick a set of nodes to remove
    node_splitter = RandomNodeSplit(num_val=0.0, num_test=val_node_ratio)
    new_data = node_splitter(data)

    if data_name == "igb-tiny":
        this_mask = torch.tensor(random.sample(range(0,data.x.size(0)), int(val_node_ratio * data.x.size(0))))
        train_mask = torch.ones([data.x.size(0)])
        test_mask = torch.zeros([data.x.size(0)])
        train_mask[this_mask] = 0
        test_mask[this_mask] = 1
        train_mask = train_mask.bool()
        test_mask = test_mask.bool()

        new_data.train_mask = train_mask
        new_data.test_mask = test_mask
    
    # Step 2: Split the edges connecting old-old nodes for training, inference and testing
    rows, cols = new_data.edge_index
    old_old_edges = create_mask(new_data.train_mask, rows, cols)
    old_old_ei = new_data.edge_index[:, old_old_edges]
    old_old_train, old_old_val, old_old_test = split_edges(old_old_ei, old_old_extra_ratio, test_ratio)

    # Step 3: Split the edges connecting old-new nodes for inference and testing
    old_new_edges = (new_data.train_mask[rows] & new_data.test_mask[cols]) | (
        new_data.test_mask[rows] & new_data.train_mask[cols])
    old_new_ei = new_data.edge_index[:, old_new_edges]
    old_new_train, _, old_new_test = split_edges(old_new_ei, 0.0, test_ratio)

    # Step 4: Split the edges connecting new-new nodes for inference and testing
    new_new_edges = create_mask(new_data.test_mask, rows, cols)
    new_new_ei = new_data.edge_index[:, new_new_edges]
    new_new_train, _, new_new_test = split_edges(new_new_ei, 0.0, test_ratio)

    # Step 5: Merge testing edges
    test_edge_index = torch.cat([old_old_test, old_new_test, new_new_test], dim=-1)
    test_edge_bundle = (old_old_test, old_new_test, new_new_test, test_edge_index)

    # Step 6: Prepare the graph for training
    training_only_ei = subgraph(new_data.train_mask, old_old_train, relabel_nodes=True)[0]
    training_only_x = new_data.x[new_data.train_mask]
 
    # Step 7: Generate training/validation set
    given_data = Data(training_only_x, training_only_ei)
    val_splitter = RandomLinkSplit(0.0, val_ratio, is_undirected=True)
    training_data, _, val_data = val_splitter(given_data)
    split_edge['train']['edge'] = training_data.edge_index.t()

    val_pos_edge_index = val_data.edge_label_index.t()[val_data.edge_label.bool()].t()

    ##### sample positive nodes for validation nodes
    valid_dict = {}
    val_row, val_col = val_pos_edge_index
    for i in range(val_row.size(0)):
        if val_row[i].item() in valid_dict:
            valid_dict[val_row[i].item()]["positive"].append(val_col[i].item())
        else:
            valid_dict[val_row[i].item()] = {}
            valid_dict[val_row[i].item()]["positive"] = [val_col[i].item()]
        if val_col[i].item() in valid_dict:
            valid_dict[val_col[i].item()]["positive"].append(val_row[i].item())
        else:
            valid_dict[val_col[i].item()] = {}
            valid_dict[val_col[i].item()]["positive"] = [val_row[i].item()]

    neighbor_pos = given_data.edge_index
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
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    ### sample 100 negative nodes for each validation node
    for key in valid_dict:
        pool = [i for i in range(given_data.x.size(0)) if i not in neighbor_index[key] and i != key]
        valid_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    valid_edges = {}
    for key in valid_dict:
        pos_line = [key for i in range(len(valid_dict[key]["positive"]))]
        neg_line = [key for i in range(len(valid_dict[key]["negative"]))]
        valid_edges[key] = {}
        valid_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(valid_dict[key]["positive"])], dim=0)
        valid_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(valid_dict[key]["negative"])], dim=0)
    split_edge['valid']['new'] = valid_edges

    # Step 8: Merge the edges for inference
    inference_edge_index = torch.cat([old_old_train, old_old_val, old_new_train, new_new_train], dim=-1)
    inference_data = Data(new_data.x, inference_edge_index)

    ##### sample positive nodes for testing nodes
    test_dict = {}
    test_row, test_col = test_edge_index
    for i in range(test_row.size(0)):
        if test_row[i].item() in test_dict:
            test_dict[test_row[i].item()]["positive"].append(test_col[i].item())
        else:
            test_dict[test_row[i].item()] = {}
            test_dict[test_row[i].item()]["positive"] = [test_col[i].item()]
        if test_col[i].item() in test_dict:
            test_dict[test_col[i].item()]["positive"].append(test_row[i].item())
        else:
            test_dict[test_col[i].item()] = {}
            test_dict[test_col[i].item()]["positive"] = [test_row[i].item()]

    neighbor_pos = data.edge_index
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
    
    for key in neighbor_index:
        neighbor_index[key] = list(set(neighbor_index[key]))

    ### sample 100 negative nodes for each testing node
    for key in test_dict:
        pool = [i for i in range(data.x.size(0)) if i not in neighbor_index[key] and i != key]
        test_dict[key]["negative"] = random.sample(pool, k=negative_samples)

    test_edges = {}
    for key in test_dict:
        pos_line = [key for i in range(len(test_dict[key]["positive"]))]
        neg_line = [key for i in range(len(test_dict[key]["negative"]))]
        test_edges[key] = {}
        test_edges[key]["positive"] = torch.stack([torch.tensor(pos_line), torch.tensor(test_dict[key]["positive"])], dim=0)
        test_edges[key]["negative"] = torch.stack([torch.tensor(neg_line), torch.tensor(test_dict[key]["negative"])], dim=0)
    split_edge['test']['new'] = test_edges

    print("Datasets Infomation:\t\n")
    print("Name:\t" + data_name + "\n")
    print("#Old Nodes:\t" + str(training_only_x.size(0))+"\n")
    print("#New Nodes:\t" + str(new_data.x.size(0) - training_only_x.size(0))+"\n")
    print("#Old-Old testing edges:\t" + str(old_old_test.size(1))+"\n")
    print("#Old-New testing edges:\t" + str(old_new_test.size(1))+"\n")
    print("#New-New testing edges:\t" + str(new_new_test.size(1))+"\n")

    split_edge_degree = "data/" + data_name + "-" + str(test_ratio * 10) + "-" + str(val_node_ratio*10) + "-" + str(val_ratio*10) + "-" + str(old_old_extra_ratio*10) + "-" + str(negative_samples) + "neg-induc_dict.json"

    #### Calculate the node degree
    neighbor_pos = old_old_train
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

    import json
    file = open(split_edge_degree, "w")
    file.write(json.dumps(edge_dict))
    file.close()

    return training_data, inference_data, split_edge

if __name__ == "__main__":
    from utils import get_dataset
    dataset_name = 'citeseer'
    dataset = get_dataset('./data', dataset_name)
    data = dataset[0]

    # split_edge = do_edge_split_with_ratio(dataset, negative_samples=500)
    # torch.save(split_edge, "data/" + dataset_name + "-500neg" + "_new.pkl")
    training_data, val_data, valid_edges, inference_data, test_edges = split_edge_inductive(data, negative_sample_num=500)