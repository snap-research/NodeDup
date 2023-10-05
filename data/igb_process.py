import argparse, dgl
from igb.dataloader import IGB260MDGLDataset
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../IGB-Datasets', help='path containing the datasets')
parser.add_argument('--dataset_size', type=str, default='tiny',choices=['tiny', 'small', 'medium', 'large', 'full'], help='size of the datasets')
parser.add_argument('--num_classes', type=int, default=19, choices=[19, 2983], help='number of classes')
parser.add_argument('--in_memory', type=int, default=0, choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
parser.add_argument('--synthetic', type=int, default=0,choices=[0, 1], help='0:nlp-node embeddings, 1:random')
args = parser.parse_args()
dataset = IGB260MDGLDataset(args)
dgl_graph = dataset[0]
node_features = dgl_graph.ndata['feat']
import torch
node_features = torch.tensor(node_features)
edges = dgl_graph.edges()
edge_index = torch.stack(edges).long()
from torch_geometric.data import Data
torch_graph = Data(x=node_features, edge_index=edge_index)
torch.save(torch_graph, "./igb-tiny.pkl")