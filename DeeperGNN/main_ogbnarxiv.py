import torch
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_scatter import scatter_add

from texttable import Texttable
import argparse

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, norm):


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
            
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()

        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
        
        
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, K, dropout):
        super(Net, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.bn = torch.nn.BatchNorm1d(hidden)
        self.prop = Prop(num_classes, K)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.norm
        x = F.relu(self.bn(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, norm)
        return F.log_softmax(x, dim=1)
    
    
### Hypyparameters
parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--K', type=int, default=16)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--log_steps', type=int, default=1)


args = parser.parse_args()


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

tab_printer(args)

###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0]
num_features = dataset.num_features
num_classes = dataset.num_classes

data.edge_index = to_undirected(data.edge_index, data.num_nodes)
data = data.to(device)


### do nomalization for only one time
data.edge_index, data.norm = gcn_norm(data.edge_index, edge_weight=None, num_nodes=data.x.size(0), dtype=data.x.dtype)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(args.runs, None)

model = Net(num_features, num_classes, args.hidden, args.K, args.dropout).to(device)

print('#Parameters:', sum(p.numel() for p in model.parameters()))

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)
        logger.add_result(run, result)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()

